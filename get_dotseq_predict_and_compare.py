import argparse
import os
import torch
import random
import subprocess
import numpy as np
import json
import shutil
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

# Import GraphArna modules
from build.lib.grapharna.preprocess_rna_pdb import get_dotseq_from_pdb
from grapharna import dot_to_bpseq, process_rna_file
from grapharna.datasets import RNAPDBDataset
from grapharna.utils import Sampler, read_dotseq_file
from grapharna.main_rna_pdb import sample
from grapharna.models import PAMNet, Config

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_lddt(model_path, ref_path):
    abs_model = os.path.abspath(model_path)
    abs_ref = os.path.abspath(ref_path)
    
    # Check if files actually exist and aren't empty (0 bytes) before trying Docker
    if not os.path.isfile(abs_model) or os.path.getsize(abs_model) == 0:
        print(f"Error: Model PDB is missing or empty at {abs_model}")
        return 0.0, {}
    if not os.path.isfile(abs_ref) or os.path.getsize(abs_ref) == 0:
        print(f"Error: Reference PDB is missing or empty at {abs_ref}")
        return 0.0, {}

    model_dir = os.path.dirname(abs_model)
    model_filename = os.path.basename(abs_model)
    ref_dir = os.path.dirname(abs_ref)
    ref_filename = os.path.basename(abs_ref)
    
    current_dir = os.path.abspath(".") 
    report_path = os.path.join(current_dir, "ost_report.json")
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{model_dir}:/model_dir",
        "-v", f"{ref_dir}:/ref_dir",
        "-v", f"{current_dir}:/outdir",  
        "registry.scicore.unibas.ch/schwede/openstructure:latest",
        "compare-structures",
        "-m", f"/model_dir/{model_filename}",
        "-r", f"/ref_dir/{ref_filename}",
        "--lddt",
        "--local-lddt",
        "--lddt-no-stereochecks",
        "-o", "/outdir/ost_report.json"
    ]
    
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                data = json.load(f)
            os.remove(report_path)
            
            global_score = data.get("lddt", 0.0)
            raw_local_scores = data.get("local_lddt", {})
            clean_local_scores = {res_id.rstrip('.'): (0.0 if score is None else score) 
                                  for res_id, score in raw_local_scores.items()}
            return global_score, clean_local_scores
        return 0.0, {}
    except subprocess.CalledProcessError as e:
        print(f"Failed to run OpenStructure:\n{e.output.decode()}")
        return 0.0, {}

# 1. Setup Arguments and Directories
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input PDB file')
parser.add_argument('--output-dir', type=str, required=True, help='Base output directory')
args = parser.parse_args()

# Exact naming required by GraphaRNA
original_input = args.input
base_name = os.path.basename(original_input)  # e.g., 5Y7M_1_D_D_23_U.pdb
dir_name = base_name.replace('.pdb', '')      # e.g., 5Y7M_1_D_D_23_U
dotseq_name = dir_name + '.dotseq'            # e.g., 5Y7M_1_D_D_23_U.dotseq

final_output_path = os.path.abspath(os.path.join(args.output_dir, dir_name))
os.makedirs(final_output_path, exist_ok=True)

print(f"--- Processing: {dir_name} ---")

# 2. Preprocessing
dotseq_file_path = os.path.join(final_output_path, dotseq_name)
with open(dotseq_file_path, 'w') as f:
    f.write(get_dotseq_from_pdb(original_input))

set_seed(0)
_, dot, seq = read_dotseq_file(dotseq_file_path)
bpseq = dot_to_bpseq(dot)

# Process RNA file directly into the final output directory
process_rna_file(
    rna_file=dotseq_file_path,
    seq_segments=seq,
    file_3d_type=".dotseq",
    sampling=True,
    save_dir_full=final_output_path, 
    name=dotseq_name, # Critical: Must have the .dotseq extension here!
    res_pairs=bpseq
)

# 3. Model Loading
exp_name = "grapharna"
model_path = f"save/{exp_name}/model_800.h5"
config = Config(dataset=None, dim=256, n_layer=6, cutoff_l=0.5, cutoff_g=1.6,
                mode='coarse-grain', knns=20, transformer_blocks=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PAMNet(config)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval().to(device)
print("Model loaded! Device:", device)

# 4. Inference
root_dir = os.path.dirname(final_output_path) # e.g., /results
ds = RNAPDBDataset(root_dir, name=dir_name, mode='coarse-grain')
print(f"Dataset length: {len(ds)} (If this is 0, the dataloader is empty!)")

ds_loader = DataLoader(ds, batch_size=8, shuffle=False)
sampler = Sampler(timesteps=1000)

predicted_pdb_name = f"{dir_name}-pred.pdb"
predicted_pdb_path = os.path.join(final_output_path, predicted_pdb_name)

print("Sampling...")
sample(model, ds_loader, device, sampler, 800, None, 
       num_batches=None, 
       exp_name=f"{exp_name}-seed=0", 
       output_folder=final_output_path, 
       output_name=predicted_pdb_name)

reference_pdb_path = os.path.join(final_output_path, f"{dir_name}_reference.pdb")
shutil.copy(original_input, reference_pdb_path)

# 5. Run Arena
print("\n--- Running Arena ---")
arena_command = ["./Arena/Arena", predicted_pdb_path, predicted_pdb_path, "5"]
try:
    subprocess.run(arena_command, check=True)
except Exception as e:
    print(f"Arena execution failed: {e}")

# 6. Calculate lDDT and Save Final Report
print("\n--- Calculating lDDT ---")
global_score, local_scores = calculate_lddt(predicted_pdb_path, reference_pdb_path)

lddt_data = {
    "global_lddt": global_score,
    "local_lddt": local_scores
}

with open(os.path.join(final_output_path, "results_summary.json"), 'w') as f:
    json.dump(lddt_data, f, indent=4)

print(f"\nDONE. All outputs are successfully stored in: {final_output_path}")