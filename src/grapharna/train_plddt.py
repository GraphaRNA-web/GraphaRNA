import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max
from torch.utils.data import random_split

from grapharna.models import PAMNet, Config, pLDDTHead
from grapharna.datasets import RNAPDBDataset
from grapharna.main_rna_pdb import set_seed 
import torch.nn.functional as F
def plddt_to_bins(plddt_scores, num_bins=50):
    bin_indices = torch.floor(plddt_scores * num_bins).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    return bin_indices


def validation(pamnet, plddt_head, loader, device):
    plddt_head.eval()
    l1_losses = []
    
    with torch.no_grad():
        for data, name, seqs in loader:
            data = data.to(device)
            is_c4_prime = data.x[:, 11].bool() 
            res_idx = is_c4_prime.cumsum(dim=0) - 1 # Dynamically groups atoms into residues
            true_plddt = data.plddt[is_c4_prime].to(device) # Safely extracts exactly 1 score per residue
            t = torch.zeros(data.batch.size(0), device=device).long()
            
            _, hidden_features = pamnet(data, seqs, t, return_hidden=True)
            
            # Predict Logits
            logits = plddt_head(hidden_features.detach(), res_idx)
            
            # Convert logits back to continuous score for evaluation metric
            pred_plddt = plddt_head.get_plddt_score(logits)
            
            res_mask = (true_plddt > 0.0)
            
            if not res_mask.any():
                continue
                
            loss = F.l1_loss(pred_plddt[res_mask], true_plddt[res_mask])
            l1_losses.append(loss.item())
            
    plddt_head.train()
    return np.mean(l1_losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train head.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to frozen PAMNet weights')
    
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='Filename for saving training logs')
    
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--cutoff_l', type=float, default=.5)
    parser.add_argument('--cutoff_g', type=float, default=1.6)
    parser.add_argument('--mode', type=str, default='coarse-grain')
    parser.add_argument('--knns', type=int, default=20)
    parser.add_argument('--blocks', type=int, default=6)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    save_folder = "./save/plddt_head_binned_weighted"
    os.makedirs(save_folder, exist_ok=True)
    log_path = os.path.join(save_folder, args.log_file)
    
    def log_message(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log_message(f"Training on Device: {device}")

    path = osp.join('.', 'data', args.dataset)
    train_dataset = RNAPDBDataset(path, name='train-pkl', mode=args.mode)
    test_dataset = RNAPDBDataset(path, name='test-pkl', mode=args.mode)
    
    

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer,
                    cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g, mode=args.mode,
                    knns=args.knns, transformer_blocks=args.blocks)
    
    pamnet = PAMNet(config).to(device)
    pamnet.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)    
    for param in pamnet.parameters():
        param.requires_grad = False
    pamnet.eval()

    input_dim = pamnet.seq_emb_dim + pamnet.dim + pamnet.total_dim

    input_dim = pamnet.seq_emb_dim + pamnet.dim + pamnet.total_dim
    plddt_head = pLDDTHead(input_dim=input_dim, num_bins=10, dropout_rate=0.3).to(device)

    log_message("Calculating bin frequencies for inverse-sqrt weighting...")
    bin_counts = torch.zeros(10, dtype=torch.float32)
    
    for data, name, seq in train_dataset:
        valid_plddt = data.plddt[::5]
        valid_plddt = valid_plddt[valid_plddt > 0.0]
        
        if len(valid_plddt) > 0:
            bins = plddt_to_bins(valid_plddt, num_bins=10)
            bin_counts += torch.bincount(bins, minlength=10).cpu()
            
    raw_weights = 1.0 / torch.sqrt(bin_counts + 1e-5)
    
    normalized_weights = raw_weights / raw_weights.mean()
    
    class_weights = normalized_weights.to(device)
    log_message(f"Computed Class Weights: {class_weights.cpu().numpy().round(4).tolist()}")

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(plddt_head.parameters(), lr=args.lr, weight_decay=1e-4)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        log_message(f"Loading checkpoint from {args.resume_checkpoint}...")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            plddt_head.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            log_message(f"Successfully resumed from epoch {start_epoch}")
        else:
            plddt_head.load_state_dict(checkpoint)
            log_message("Loaded model weights only. Starting from epoch 0.")

    log_message("Start Training pLDDT Head!")
    
    for epoch in range(start_epoch, args.epochs):
        plddt_head.train()
        train_losses = []
        step = 0
        
        for data, name, seqs in train_loader:
            data = data.to(device)
            is_c4_prime = data.x[:, 11].bool() 
            res_idx = is_c4_prime.cumsum(dim=0) - 1 # Dynamically groups atoms into residues
            true_plddt = data.plddt[is_c4_prime].to(device) # Safely extracts exactly 1 score per residue
            t = torch.zeros(data.batch.size(0), device=device).long()
            
            optimizer.zero_grad()

            with torch.no_grad():
                _, hidden_features = pamnet(data, seqs, t, return_hidden=True)
            
            detached_features = hidden_features.detach()
            
            logits = plddt_head(detached_features, res_idx)
            res_mask = (true_plddt > 0.0)

            if not res_mask.any():
                log_message(f"Warning: Batch {step} has no valid pLDDT scores (all 0.0). Skipping.")
                continue

            target_bins = plddt_to_bins(true_plddt, num_bins=10)

            loss = ce_loss_fn(logits[res_mask], target_bins[res_mask])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(plddt_head.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if step % 20 == 0 and step != 0:
                log_message(f"Epoch: {epoch}, Step: {step}, Train Loss (CE): {np.mean(train_losses):.4f}")
            step += 1
            
        val_l1_error = validation(pamnet, plddt_head, test_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        log_message(f'*** Epoch: {epoch+1} Completed | Train Loss (CE): {np.mean(train_losses):.4f} | Val L1 Error: {val_l1_error:.4f} | LR: {current_lr:.6f} ***')
        
        scheduler.step(val_l1_error)
        
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': plddt_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        checkpoint_path = f"{save_folder}/plddt_head_epoch_{epoch+1}.h5"
        torch.save(checkpoint_dict, checkpoint_path)

if __name__ == "__main__":
    main()