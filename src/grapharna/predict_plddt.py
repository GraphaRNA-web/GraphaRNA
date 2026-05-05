import os
import os.path as osp
import argparse
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
from torch_geometric.loader import DataLoader

from grapharna.models import PAMNet, Config, pLDDTHead
from grapharna.datasets import RNAPDBDataset
from grapharna.preprocess_rna_pdb import construct_graphs

def insert_plddt_to_pdb(input_pdb_path, output_pdb_path, plddt_scores):
    """
    Reads a PDB file and rewrites it, replacing the B-factor column (cols 61-66) 
    with the predicted pLDDT scores (scaled 0-100) for RNA residues.
    """
    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()
    
    out_lines = []
    res_idx = -1
    last_res_id = None
    
    for line in lines:
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            res_name = line[17:20].strip()
            
            # The dataset only extracts RNA residues (A, C, G, U). 
            if res_name in ['A', 'C', 'G', 'U', 'T']:
                chain = line[21]
                res_seq = line[22:26]
                icode = line[26]
                current_res_id = (chain, res_seq, icode)
                
                # If we've moved to a new residue, advance our score pointer
                if current_res_id != last_res_id:
                    res_idx += 1
                    last_res_id = current_res_id
                
                # Grab the score, multiply by 100 (standard for PDB pLDDT), and format to 6 chars
                if res_idx < len(plddt_scores):
                    score = plddt_scores[res_idx] * 100.0
                else:
                    score = 0.0 # Fallback if there's a mismatch
                    
                score_str = f"{score:6.2f}"
                
                # Inject the score into columns 61-66 (0-indexed 60:66)
                line = line[:60] + score_str + line[66:]
                
        out_lines.append(line)
        
    with open(output_pdb_path, 'w') as f:
        f.writelines(out_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    
    # Input options for raw PDBs
    parser.add_argument('--input_pdb', type=str, default=None, help='Path to a single PDB file to predict')
    parser.add_argument('--pdb_dir', type=str, default=None, help='Directory of PDB files to predict')
    parser.add_argument('--seq_dir', type=str, default=None, help='(Optional) Directory of corresponding .seq files')
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    
    # Model checkpoints
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to frozen PAMNet weights')
    parser.add_argument('--plddt_weights', type=str, required=True, help='Path to trained pLDDT head weights')
    
    # PAMNet config args (must match what was used during training)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--cutoff_l', type=float, default=5)
    parser.add_argument('--cutoff_g', type=float, default=16)
    parser.add_argument('--mode', type=str, default='coarse-grain')
    parser.add_argument('--knns', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=6)
    
    # Output parameters
    parser.add_argument('--output_csv', type=str, default='plddt_predictions.csv', help='Where to save results CSV')
    parser.add_argument('--output_pdb_dir', type=str, default='plddt_pdbs', help='Directory to save modified PDB files')
    args = parser.parse_args()

    if not args.input_pdb and not args.pdb_dir:
        raise ValueError("You must provide either --input_pdb or --pdb_dir")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Running Inference on Device: {device}")

    os.makedirs(args.output_pdb_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_base:
        
        # --- 1. Preprocess Raw PDBs on the fly ---
        print("Preprocessing PDB file(s) into graphs...")
        pdbs_dir = args.pdb_dir
        
        if args.input_pdb:
            pdbs_dir = osp.join(temp_base, "input_pdbs")
            os.makedirs(pdbs_dir, exist_ok=True)
            shutil.copy(args.input_pdb, pdbs_dir)
            
        save_name = "pred-pkl"
        
        construct_graphs(seq_dir=args.seq_dir, 
                         pdbs_dir=pdbs_dir, 
                         natives_dir=None, 
                         save_dir=temp_base, 
                         save_name=save_name, 
                         file_3d_type='.pdb', 
                         extended_dotbracket=False, 
                         sampling=False)
        
        # --- 2. Load Dataset ---
        test_dataset = RNAPDBDataset(temp_base, name=save_name, mode=args.mode)
        
        if len(test_dataset) == 0:
            print("Error: No valid PDB files were successfully processed.")
            return
            
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # --- 3. Initialize Models ---
        config = Config(dataset='inference', dim=args.dim, n_layer=args.n_layer,
                        cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g, mode=args.mode,
                        knns=args.knns, transformer_blocks=args.blocks)
        
        pamnet = PAMNet(config).to(device)
        pamnet.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)
        pamnet.eval() 

        input_dim = pamnet.seq_emb_dim + pamnet.dim + pamnet.total_dim
        plddt_head = pLDDTHead(input_dim=input_dim).to(device)
        plddt_head.load_state_dict(torch.load(args.plddt_weights, map_location=device)["model_state_dict"])
        plddt_head.eval()

        print("Models loaded successfully. Starting inference...")
        results = []

        # --- 4. Inference Loop ---
        with torch.no_grad():
            for data, names, seqs in test_loader:
                data = data.to(device)
                
                res_idx = (torch.arange(data.x.size(0), device=device) // 5)
                
                t = torch.zeros(data.batch.size(0), device=device).long()
                _, hidden_features = pamnet(data, seqs, t, return_hidden=True)
                pred_plddt = plddt_head(hidden_features, res_idx).cpu().numpy()
                
                residue_counts = torch.bincount(data.batch[::5]).cpu().numpy()
                
                start_idx = 0
                for i, num_res in enumerate(residue_counts):
                    end_idx = start_idx + num_res
                    
                    rna_name = names[i]
                    seq = seqs[i]
                    preds_for_rna = pred_plddt[start_idx:end_idx]
                    
                    for res_pos in range(num_res):
                        results.append({
                            "RNA_Name": rna_name,
                            "Residue_Index": res_pos + 1,
                            "Nucleotide": seq[res_pos] if res_pos < len(seq) else "?",
                            "Predicted_pLDDT": round(float(preds_for_rna[res_pos]), 4)
                        })
                    
                    original_pdb_path = osp.join(pdbs_dir, rna_name)
                    new_pdb_name = rna_name.replace(".pdb", "_plddt.pdb")
                    output_pdb_path = osp.join(args.output_pdb_dir, new_pdb_name)
                    
                    insert_plddt_to_pdb(original_pdb_path, output_pdb_path, preds_for_rna)
                    
                    start_idx = end_idx

        print(f"\nInference complete! Results saved to {args.output_csv}")
        print(f"PDBs with injected pLDDT scores saved to: {args.output_pdb_dir}/")

if __name__ == "__main__":
    main()