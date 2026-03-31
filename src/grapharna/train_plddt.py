import os
import os.path as osp
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from grapharna.models import PAMNet, Config
from grapharna.datasets import RNAPDBDataset
from grapharna.main_rna_pdb import set_seed # Reusing your seed function

# ==========================================
# 1. Define the pLDDT Head
# ==========================================
class pLDDTHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(pLDDTHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Assuming target pLDDT is normalized between 0 and 1
        )

    def forward(self, hidden_features):
        return self.mlp(hidden_features).squeeze(-1)

# ==========================================
# 2. Validation Function
# ==========================================
def validation(pamnet, plddt_head, loader, device, loss_fn):
    plddt_head.eval()
    losses = []
    
    with torch.no_grad():
        for data, name, seqs in loader:
            data = data.to(device)
            mask = data.x[:, -1].bool() #
            
            # Ground truth pLDDT (Update this based on your dataset implementation)
            true_plddt = data.plddt.to(device) 
            
            # We evaluate on the clean structure, so timestep t is 0
            t = torch.zeros(data.batch.size(0), device=device).long()
            
            # Extract features from frozen PAMNet
            _, hidden_features = pamnet(data, seqs, t)
            
            # Predict pLDDT
            pred_plddt = plddt_head(hidden_features)
            
            # Calculate Loss on masked atoms
            loss = loss_fn(pred_plddt[mask], true_plddt[mask])
            losses.append(loss.item())
            
    plddt_head.train()
    return np.mean(losses)

# ==========================================
# 3. Main Training Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=40, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='RNA-Puzzles', help='Dataset to be used')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train head.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to frozen PAMNet weights')
    
    # PAMNet config args
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--cutoff_l', type=float, default=5)
    parser.add_argument('--cutoff_g', type=float, default=16)
    parser.add_argument('--mode', type=str, default='coarse-grain')
    parser.add_argument('--knns', type=int, default=2)
    parser.add_argument('--blocks', type=int, default=4)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    print(f"Training on Device: {device}")

    # --- Load Datasets ---
    path = osp.join('.', 'data', args.dataset)
    train_dataset = RNAPDBDataset(path, name='train-pkl', mode=args.mode).shuffle()
    val_dataset = RNAPDBDataset(path, name='val-pkl', mode=args.mode)
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Initialize & Freeze PAMNet ---
    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer,
                    cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g, mode=args.mode,
                    knns=args.knns, transformer_blocks=args.blocks)
    
    pamnet = PAMNet(config).to(device)
    pamnet.load_state_dict(torch.load(args.pretrained_model, map_location=device), strict=False)    
    for param in pamnet.parameters():
        param.requires_grad = False
    pamnet.eval() # Important: keeps LayerNorm/Dropout static

    # --- Initialize pLDDT Head ---
    input_dim = pamnet.seq_emb_dim + pamnet.dim + pamnet.total_dim
    plddt_head = pLDDTHead(input_dim=input_dim).to(device)

    # --- Setup Optimizer & Loss ---
    optimizer = optim.Adam(plddt_head.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Use MSE or L1 loss for regression tasks like pLDDT
    loss_fn = nn.MSELoss() 
    
    print("Start Training pLDDT Head!")
    
    for epoch in range(args.epochs):
        plddt_head.train()
        losses = []
        step = 0
        
        for data, name, seqs in train_loader:
            data = data.to(device)
            mask = data.x[:, -1].bool()
            
            # Ground truth pLDDT (Update this based on your dataset implementation)
            true_plddt = data.plddt.to(device)
            
            # Provide t=0 for native structures
            t = torch.zeros(data.batch.size(0), device=device).long()
            
            optimizer.zero_grad()

            # 1. Forward pass through frozen PAMNet (no gradients needed)
            with torch.no_grad():
                _, hidden_features = pamnet(data, seqs, t)
            
            # 2. Forward pass through pLDDT Head
            pred_plddt = plddt_head(hidden_features)

            # 3. Calculate Loss and Backpropagate
            loss = loss_fn(pred_plddt[mask], true_plddt[mask])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(plddt_head.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 20 == 0 and step != 0:
                print(f"Epoch: {epoch}, Step: {step}, Train Loss: {np.mean(losses):.4f}")
            step += 1
            
        scheduler.step()
        
        # Validation Phase
        val_loss = validation(pamnet, plddt_head, val_loader, device, loss_fn)
        print(f'*** Epoch: {epoch+1} Completed | Train Loss: {np.mean(losses):.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]} ***')
        
        # Save Head Checkpoints
        save_folder = f"./save/plddt_head"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        if (epoch + 1) % 5 == 0:
            torch.save(plddt_head.state_dict(), f"{save_folder}/plddt_head_epoch_{epoch+1}.h5")

if __name__ == "__main__":
    main()