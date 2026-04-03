import time
import torch
import torch.nn.functional as F
from tqdm import tqdm


try:
    from .dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
except ImportError:
    print("WARNING: dpm_solver_pytorch.py not found in utils. Fallback to DDPM sampling.")

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def generate_per_residue_noise(x_data, eps=1e-3):
    x_start = x_data.x.contiguous()
    atoms = x_data.x[:, -4:].sum(dim =1)
    c_n_atoms = torch.where(atoms == 1)[0].to(x_start.device)
    p_atoms = torch.where(atoms == 0)[0].to(x_start.device)
    per_residue_noise = torch.rand((c_n_atoms.shape[0])//4, x_start.shape[1], device=x_start.device) # generate noise for each C4' atom
    per_residue_noise = torch.repeat_interleave(per_residue_noise, 4, dim=0) # repeat it for all atoms in residue (except for P)
    noise = torch.zeros_like(x_start)
    noise[c_n_atoms] = per_residue_noise
    diff = torch.arange(0, len(p_atoms), device=x_start.device)
    relative_c4p = p_atoms - diff # compute the index of each C4' for every P atom
    noise[p_atoms] = noise[c_n_atoms[relative_c4p]] # if there is a P atom, copy the noise from the corresponding C4' atom
    noise = noise + torch.randn_like(x_start, device=x_start.device) * eps

    return noise

class DPMSolverWrapper:
    """
    Bridges DPM-Solver with GraphaRNA graph structure.
    Extracts 3D coordinates from the graph object and processes noise.
    """
    def __init__(self, model, seqs, context_mols, coord_mask, num_train_timesteps):
        self.model = model
        self.seqs = seqs
        self.context_mols = context_mols
        self.coord_mask = coord_mask
        self.num_train_timesteps = num_train_timesteps
        self.device = next(model.parameters()).device
        self.num_nodes = context_mols.x.shape[0]
        self.visited_timesteps = []

    def _to_discrete_timestep(self, t):
        # DPM-Solver wrapper passes model time in [0, 1000] for discrete schedules.
        # Convert it back to the training index range [0, num_train_timesteps - 1].
        if torch.max(t) <= 1.0 + 1e-6:
            t_discrete = ((t - 1.0 / self.num_train_timesteps) * self.num_train_timesteps).round().to(torch.long)
        else:
            t_discrete = (t * self.num_train_timesteps / 1000.0).round().to(torch.long)
        return torch.clamp(t_discrete, 0, self.num_train_timesteps - 1)

    def __call__(self, x, t):
        t_discrete = self._to_discrete_timestep(t)
        self.visited_timesteps.append(int(t_discrete[0].item()))
        
        t_discrete = torch.full((self.num_nodes,), t_discrete[0].item(), device=self.device, dtype=torch.long)

        atoms_mask = 1 - self.coord_mask
        
        # Inject noisy coords into the PyG graph object
        self.context_mols.x = x * self.coord_mask + self.context_mols.x * atoms_mask

        # Predict noise using the GNN model
        predicted_noise = self.model(self.context_mols, self.seqs, t_discrete)

        # Return only the predicted noise applied to 3D coordinates
        return predicted_noise * self.coord_mask

class Sampler():
    def __init__(self, timesteps: int, channels: int=3, use_dpm_solver: bool=False, dpm_steps: int=20, mode: str='ddpm', dpm_skip_type: str='time_quadratic'):
        self.timesteps = timesteps
        self.channels = channels
        self.use_dpm_solver = use_dpm_solver
        self.dpm_steps = dpm_steps
        self.mode = mode
        self.dpm_skip_type = dpm_skip_type
        
        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0) # Saved for DPM-Solver
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    @torch.no_grad()
    def p_sample(self, model, seqs, x_raw, t, t_index, coord_mask, atoms_mask):
        x = x_raw.x * coord_mask
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x_raw, seqs, t)*coord_mask / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            x_raw.x = model_mean * coord_mask + x_raw.x * atoms_mask
            return x_raw.x
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            out = model_mean + torch.sqrt(posterior_variance_t) * noise
            x_raw.x = out * coord_mask + x_raw.x * atoms_mask
            return x_raw.x


    def add_fixed(self, raw_x, fixed, t, t_index, x_start):
        if torch.any(fixed) and t_index > 0:
            denoised_raw = self.q_sample(x_start, t - 1)
            raw_x[fixed] = denoised_raw[fixed]
        if torch.any(fixed) and t_index == 0:
            raw_x[fixed] = x_start[fixed]
        return raw_x
 
    # Algorithm 2
    @torch.no_grad()
    def p_sample_loop(self, model, seqs, shape, context_mols):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        noise = torch.randn_like(context_mols.x, device=device)
        denoised = []
        
        context_mols.x = noise * coord_mask + context_mols.x * atoms_mask
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            context_mols.x = self.p_sample(model, seqs, context_mols, torch.full((b,), i, device=device, dtype=torch.long), i, coord_mask, atoms_mask)
            # denoised.append(context_mols.clone().cpu())
        denoised.append(context_mols.clone().cpu())
        return denoised

    # ------------------------------------------------------------------
    # DPM-SOLVER LOOP (Replaces p_sample_loop)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def dpm_sample_loop(self, model, seqs, shape, context_mols):
        device = next(model.parameters()).device
        
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        
        x_T = torch.randn_like(context_mols.x, device=device) * coord_mask
        
        # Configure noise schedule for DPM using discrete alphas
        noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod.to(device))
        if noise_schedule.total_N != self.timesteps:
            raise RuntimeError(
                f"Noise schedule length mismatch: solver total_N={noise_schedule.total_N}, "
                f"expected timesteps={self.timesteps}. This would desynchronize training and inference schedules."
            )

        model_fn = DPMSolverWrapper(model, seqs, context_mols, coord_mask, num_train_timesteps=self.timesteps)
        
        wrapped_model = model_wrapper(
            model_fn,
            noise_schedule,
            model_type="noise",
            model_kwargs={},
        )
        
        dpm_solver = DPM_Solver(wrapped_model, noise_schedule, algorithm_type="dpmsolver++")

        # Inspect planned timestep trajectory before solving.
        planned_t = dpm_solver.get_time_steps(
            skip_type=self.dpm_skip_type,
            t_T=noise_schedule.T,
            t_0=1.0 / noise_schedule.total_N,
            N=self.dpm_steps,
            device=device,
        )
        planned_model_t = (planned_t - 1.0 / noise_schedule.total_N) * 1000.0
        planned_discrete = model_fn._to_discrete_timestep(planned_model_t)
        planned_abs_deltas = torch.abs(planned_discrete[1:] - planned_discrete[:-1])

        print(
            f"DPM planned time grid ({self.dpm_skip_type}): "
            f"points={planned_t.numel()}, first_t={planned_t[0].item():.6f}, last_t={planned_t[-1].item():.6f}"
        )
        print(
            "DPM planned mapped indices: "
            f"min={planned_discrete.min().item()}, max={planned_discrete.max().item()}, "
            f"unique={torch.unique(planned_discrete).numel()}"
        )
        print(
            "DPM planned mapped indices (head/tail): "
            f"head={planned_discrete[:min(10, planned_discrete.numel())].tolist()} "
            f"tail={planned_discrete[-min(10, planned_discrete.numel()):].tolist()}"
        )
        if planned_abs_deltas.numel() > 0:
            delta_values, delta_counts = torch.unique(planned_abs_deltas, return_counts=True)
            top_k = min(10, delta_values.numel())
            top_vals = delta_values[:top_k].tolist()
            top_cnts = delta_counts[:top_k].tolist()
            print(
                "DPM planned |delta index| stats: "
                f"min={planned_abs_deltas.min().item()}, max={planned_abs_deltas.max().item()}, "
                f"mean={planned_abs_deltas.float().mean().item():.2f}"
            )
            print(f"DPM planned |delta index| histogram (smallest {top_k} bins): {list(zip(top_vals, top_cnts))}")

        print(f"Starting DPM-Solver++ sampling ({self.dpm_steps} steps)...")

        start_time = time.time()

        x_0 = dpm_solver.sample(
            x_T,
            steps=self.dpm_steps,
            order=2,
            skip_type=self.dpm_skip_type,
            method="multistep",
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Sampling complete in: {elapsed:.2f} seconds.")

        if model_fn.visited_timesteps:
            visited = torch.tensor(model_fn.visited_timesteps, device=device)
            visited_unique = torch.unique(visited)
            print(
                "DPM actual model calls: "
                f"nfe={visited.numel()}, unique_t={visited_unique.numel()}, "
                f"min={visited.min().item()}, max={visited.max().item()}"
            )
            print(
                "DPM actual timestep trace (head/tail): "
                f"head={visited[:min(20, visited.numel())].tolist()} "
                f"tail={visited[-min(20, visited.numel()):].tolist()}"
            )
            if visited.numel() > 1:
                visited_abs_deltas = torch.abs(visited[1:] - visited[:-1])
                print(
                    "DPM actual |delta index| stats: "
                    f"min={visited_abs_deltas.min().item()}, max={visited_abs_deltas.max().item()}, "
                    f"mean={visited_abs_deltas.float().mean().item():.2f}"
                )
                if visited_unique.numel() < max(8, self.dpm_steps // 4):
                    print(
                        "WARNING: Very few unique timesteps were visited by the model. "
                        "This can make different --steps settings look identical."
                    )
        
        # Recover final coordinates
        context_mols.x = x_0 * coord_mask + context_mols.x * atoms_mask
        
        return [context_mols.clone().cpu()]
    
    @torch.no_grad()
    def topology_aware_dpm_sample(self, model, seqs, shape, context_mols, steps=100):
        device = next(model.parameters()).device
        
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        
        # Inicjalizacja szumu
        context_mols.x = torch.randn_like(context_mols.x, device=device) * coord_mask + context_mols.x * atoms_mask
        
        # Wybór kroków (np. krok kwadratowy, aby na końcu mieć gęstszą siatkę)
        timesteps = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        
        for i in tqdm(range(len(timesteps) - 1), desc="Topology-Aware DPM"):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # 1. KROK PREDICTORA: Oblicz szum z obecnej struktury
            # Predykcja odbywa się na aktualnym grafie knn
            # 1. KROK PREDICTORA
            noise_pred = model(context_mols, seqs, t_tensor) * coord_mask
            
            # Bezpieczne wyciągnięcie alf z zabezpieczeniem numerycznym
            alpha_t = self.alphas_cumprod[t] + 1e-8
            alpha_t_next = self.alphas_cumprod[t_next]
            
            # Predykcja wektora x_0 (zabezpieczone przed eksplodującymi gradientami)
            x_0_pred = (context_mols.x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Zapobieganie gigantycznym skokom (opcjonalnie, ale stabilizuje GNN)
            x_0_pred = torch.clamp(x_0_pred, min=-50.0, max=50.0) 
            
            x_next = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
            
            # 2. KROK KOREKTORA (Topology Stabilization)
            max_disp = 1.0 
            disp = (x_next - context_mols.x) * coord_mask
            disp_norm = torch.norm(disp, dim=-1, keepdim=True)
            
            # FIX: Dodano 1e-8 aby zapobiec NaN (0/0) na końcu dyfuzji
            disp = torch.where(disp_norm > max_disp, disp / (disp_norm + 1e-8) * max_disp, disp)
            
            context_mols.x = (context_mols.x + disp) * coord_mask + context_mols.x * atoms_mask

        return [context_mols.clone().cpu()]

    @torch.no_grad()
    def sample(self, model, seqs, context_mols):
        if self.use_dpm_solver:
            if self.mode == 'custom':
                return self.topology_aware_dpm_sample(model, seqs, shape=context_mols.x.shape, 
                                                      context_mols=context_mols, steps=self.dpm_steps)
            else:
                # DPM-Solver++ (wrapper)
                return self.dpm_sample_loop(model, seqs, shape=context_mols.x.shape, context_mols=context_mols)
        else:
            return self.p_sample_loop(model, seqs, shape=context_mols.x.shape, context_mols=context_mols)

    # forward diffusion (using the nice property)
    def q_sample(self,
                 x_start,
                 t,
                 noise=None
                 ):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)