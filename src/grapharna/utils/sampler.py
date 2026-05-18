import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import knn

try:
    from .dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
except ImportError:
    print("WARNING: dpm_solver_pytorch.py not found in utils. Fallback to DDPM sampling.")

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
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
    atoms = x_data.x[:, -4:].sum(dim=1)
    c_n_atoms = torch.where(atoms == 1)[0].to(x_start.device)
    p_atoms = torch.where(atoms == 0)[0].to(x_start.device)
    
    # Generate noise for each C4' atom
    per_residue_noise = torch.rand((c_n_atoms.shape[0]) // 4, x_start.shape[1], device=x_start.device) 
    per_residue_noise = torch.repeat_interleave(per_residue_noise, 4, dim=0) 
    
    noise = torch.zeros_like(x_start)
    noise[c_n_atoms] = per_residue_noise
    
    diff = torch.arange(0, len(p_atoms), device=x_start.device)
    relative_c4p = p_atoms - diff 
    noise[p_atoms] = noise[c_n_atoms[relative_c4p]] 
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
    def __init__(self, timesteps: int, channels: int=3, use_dpm_solver: bool=False, 
                 dpm_steps: int=50, mode: str='ddpm', dpm_skip_type: str='time_quadratic',
                 ddpm_relax: int=750, dpm_method: str="singlestep", dpm_order: int=3):
        self.timesteps = timesteps
        self.channels = channels
        self.use_dpm_solver = use_dpm_solver
        self.dpm_steps = dpm_steps
        self.mode = mode
        
        # Solver hyperparameters
        self.dpm_skip_type = dpm_skip_type
        self.ddpm_relax = ddpm_relax
        self.dpm_method = dpm_method
        self.dpm_order = dpm_order

        # Define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # Define alphas 
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, model, seqs, x_raw, t, t_index, coord_mask, atoms_mask):
        x = x_raw.x * coord_mask
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x_raw, seqs, t) * coord_mask / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            x_raw.x = model_mean * coord_mask + x_raw.x * atoms_mask
            return x_raw.x
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
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

    @torch.no_grad()
    def p_sample_loop(self, model, seqs, shape, context_mols):
        device = next(model.parameters()).device
        b = shape[0]
        
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        noise = torch.randn_like(context_mols.x, device=device)
        denoised = []

        context_mols.x = noise * coord_mask + context_mols.x * atoms_mask
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Standard DDPM sampling', total=self.timesteps):
            context_mols.x = self.p_sample(model, seqs, context_mols, torch.full((b,), i, device=device, dtype=torch.long), i, coord_mask, atoms_mask)

        denoised.append(context_mols.clone().cpu())
        return denoised

    # ------------------------------------------------------------------
    # DPM-SOLVER HYBRID LOOP (0-50-750 Architecture)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def hybrid_sample(self, model, seqs, shape, context_mols):
        """
        Adaptive Hybrid Sampler: DPM-Solver++ -> DDPM Relaxation.
        Jumps directly from pure noise using ODE, then relaxes using stochastic DDPM.
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # 1. Setup masks for coordinates
        coord_mask = torch.ones_like(context_mols.x)
        coord_mask[:, 3:] = 0
        atoms_mask = 1 - coord_mask
        
        # 2. Start from pure Gaussian noise
        noise = torch.randn_like(context_mols.x, device=device)
        context_mols.x = noise * coord_mask + context_mols.x * atoms_mask
        
        # Failsafe in case DDPM relax steps exceed total timesteps
        if self.ddpm_relax >= self.timesteps:
            print("WARNING: Relaxation steps exceed total timesteps. Falling back to full DDPM.")
            return self.p_sample_loop(model, seqs, shape, context_mols)

        # Intersection points on the timeline
        t_start_dpm = self.timesteps 
        t_end_dpm = self.ddpm_relax
        
        # =================================================================
        # Phase 1: DPM-Solver++ (Fast jump through noise)
        # =================================================================
        print(f"Phase 1: DPM-Solver++ Jump (t={t_start_dpm} -> {t_end_dpm}) | Steps: {self.dpm_steps} | Order: {self.dpm_order} | Method: {self.dpm_method}")
        
        noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.alphas_cumprod.to(device))
        model_fn = DPMSolverWrapper(model, seqs, context_mols, coord_mask, num_train_timesteps=self.timesteps)
        
        wrapped_model = model_wrapper(
            model_fn,
            noise_schedule,
            model_type="noise",
        )
        
        dpm_solver = DPM_Solver(wrapped_model, noise_schedule, algorithm_type="dpmsolver++")
        
        # Map indices to continuous time [0, 1] for DPM-Solver
        t_start_continuous = t_start_dpm / self.timesteps 
        t_end_continuous = t_end_dpm / self.timesteps
        
        x_current = context_mols.x * coord_mask
        x_dpm = dpm_solver.sample(
            x_current,
            steps=self.dpm_steps, 
            t_start=t_start_continuous,
            t_end=t_end_continuous,
            order=self.dpm_order,
            skip_type=self.dpm_skip_type,
            method=self.dpm_method,
            denoise_to_zero=False # Crucial: do not remove all noise, DDPM needs it
        )
        
        # Update coordinates to the intermediate state
        context_mols.x = x_dpm * coord_mask + context_mols.x * atoms_mask

        # =================================================================
        # Phase 2: DDPM (Final relaxation and collision removal)
        # =================================================================
        print(f"Phase 2: DDPM Final Relaxation for {self.ddpm_relax} steps (t={t_end_dpm-1} -> 0)")
        for i in tqdm(reversed(range(0, t_end_dpm)), desc='DDPM Relaxation', total=self.ddpm_relax):
            t_batch = torch.full((b,), i, device=device, dtype=torch.long)
            context_mols.x = self.p_sample(model, seqs, context_mols, t_batch, i, coord_mask, atoms_mask)

        return [context_mols.clone().cpu()]

    @torch.no_grad()
    def sample(self, model, seqs, context_mols):
        if self.use_dpm_solver:
            if self.mode == 'custom':
                return self.hybrid_sample(model, seqs, shape=context_mols.x.shape, context_mols=context_mols)
            else:
                # In case you ever want pure DPM-Solver all the way to 0
                print("WARNING: Using pure DPM-Solver. Hydrogen bonds (INF) might degrade.")
                self.ddpm_relax = 0
                return self.hybrid_sample(model, seqs, shape=context_mols.x.shape, context_mols=context_mols)
        else:
            return self.p_sample_loop(model, seqs, shape=context_mols.x.shape, context_mols=context_mols)

    # Forward diffusion (q_sample)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)