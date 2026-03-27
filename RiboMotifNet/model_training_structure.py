import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# 引入混合精度训练模块
from torch.cuda.amp import autocast, GradScaler 
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, 
                             precision_score, recall_score, roc_auc_score, 
                             auc, precision_recall_curve)

# ================= 1. 全局配置 (开启加速开关) =================
class Config:
    DATA_PATH = "RiboMotif_Training_Set_V3.tsv" 
    OUTPUT_DIR = "training_logs"
    
    MAX_LEN = 80       
    VOCAB = {'<PAD>': 0, 'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}
    
    EMBED_DIM = 64      
    HIDDEN_DIM = 128    
    
    BATCH_SIZE = 4096     # [加速技巧1] 开启AMP后显存变小，Batch Size可以调大一倍(64->128)，大幅减少迭代次数
    LEARNING_RATE = 4e-3
    WEIGHT_DECAY = 1e-4 
    EPOCHS = 50          
    PATIENCE = 10        
    
    # [加速技巧2] 设置读取进程数
    # Windows下建议设为 0 或 2；Linux服务器建议设为 4 或 8
    NUM_WORKERS = 8      
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 数据集类 (保持新逻辑) =================
class RNADataset(Dataset):
    def __init__(self, tsv_file, max_len=80, vocab=None):
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.max_len = max_len
        self.vocab = vocab
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def encode_sequence(self, seq):
        seq = str(seq)[:self.max_len]
        indices = [self.vocab.get(char, 5) for char in seq]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def dbn_to_adj(self, dbn):
        # 你的 { [ 逻辑完全保留
        dbn = str(dbn)
        length = min(len(dbn), self.max_len)
        adj = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        
        for i in range(length - 1):
            adj[i, i+1] = 1.0; adj[i+1, i] = 1.0
            
        stacks = {'(': [], '[': [], '{': []}
        pair_map = {')': '(', ']': '[', '}': '{'}

        for i, char in enumerate(dbn[:length]):
            if char in stacks:
                stacks[char].append(i)
            elif char in pair_map:
                opener = pair_map[char]
                if stacks[opener]:
                    j = stacks[opener].pop()
                    adj[i, j] = 1.0; adj[j, i] = 1.0
        
        np.fill_diagonal(adj, 1.0)
        row_sum = np.sum(adj, axis=1)
        row_sum[row_sum == 0] = 1 
        adj = adj / row_sum[:, np.newaxis]
        return torch.tensor(adj, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return self.encode_sequence(row['sequence']), self.dbn_to_adj(row['structure']), self.labels[idx]

# ================= 3. 模型架构 (双塔 + Attention，完全不变) =================
class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(feature_dim, 1)
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1) 
        return torch.sum(x * weights, dim=1), weights

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, adj):
        support = self.linear(x)       
        output = torch.bmm(adj, support) 
        return self.dropout(F.relu(output))

class RiboMotifNet(nn.Module):
    def __init__(self, use_structure=True):
        super(RiboMotifNet, self).__init__()
        self.use_structure = use_structure
        self.embedding = nn.Embedding(len(Config.VOCAB), Config.EMBED_DIM, padding_idx=0)
        
        self.cnn1 = nn.Conv1d(Config.EMBED_DIM, Config.HIDDEN_DIM, 3, padding=1)
        self.cnn2 = nn.Conv1d(Config.HIDDEN_DIM, Config.HIDDEN_DIM, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(Config.HIDDEN_DIM)
        self.bn2 = nn.BatchNorm1d(Config.HIDDEN_DIM)
        
        if self.use_structure:
            self.gcn1 = GraphConvLayer(Config.EMBED_DIM, Config.HIDDEN_DIM)
            self.gcn2 = GraphConvLayer(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
        
        feat_dim = Config.HIDDEN_DIM * 2 if use_structure else Config.HIDDEN_DIM
        self.feature_proj = nn.Linear(feat_dim, Config.HIDDEN_DIM)
        self.attention = AttentionLayer(Config.HIDDEN_DIM)
        self.classifier = nn.Sequential(nn.Linear(Config.HIDDEN_DIM, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, seq, adj):
        x_emb = self.embedding(seq)
        x_cnn = x_emb.permute(0, 2, 1) 
        x_cnn = F.relu(self.bn1(self.cnn1(x_cnn)))
        x_cnn = F.relu(self.bn2(self.cnn2(x_cnn))).permute(0, 2, 1) 
        
        if self.use_structure:
            x_gcn = self.gcn2(self.gcn1(x_emb, adj), adj) 
            combined = torch.cat([x_cnn, x_gcn], dim=2) 
        else:
            combined = x_cnn 
            
        return self.classifier(self.attention(F.relu(self.feature_proj(combined)))[0])

# ================= 4. 训练引擎 (AMP加速版 + 全记录) =================

def compute_metrics(y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    try:
        roc = roc_auc_score(y_true, y_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall_curve, precision_curve)
    except:
        roc = 0.5; pr_auc = 0.0
        
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'AUROC': roc,
        'AUPRC': pr_auc
    }

def run_experiment():
    if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
    print(f"Set up output directory: {Config.OUTPUT_DIR}")
    
    # [加速技巧3] 启用 cudnn benchmark (针对固定输入尺寸优化)
    torch.backends.cudnn.benchmark = True
    
    # 1. 数据准备
    print("Loading Dataset...")
    dataset = RNADataset(Config.DATA_PATH, max_len=Config.MAX_LEN, vocab=Config.VOCAB)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # [加速技巧4] pin_memory=True 加速 CPU->GPU 传输
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 2. 模型准备
    model = RiboMotifNet(use_structure=True).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # 动态权重
    pos_ratio = dataset.labels.sum() / len(dataset)
    neg_ratio = 1 - pos_ratio
    pos_weight = torch.tensor([neg_ratio / pos_ratio]).to(Config.DEVICE) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # [加速技巧5] 初始化 AMP Scaler
    scaler = GradScaler()

    # 3. 日志准备
    log_df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'AUROC', 'AUPRC', 'Time'])
    csv_path = os.path.join(Config.OUTPUT_DIR, "training_log.csv")
    
    print("\n" + "="*85)
    print(f"{'Epoch':^5} | {'Tr Loss':^8} | {'Va Loss':^8} | {'Acc':^6} | {'F1':^6} | {'MCC':^6} | {'AUC':^6} | {'Time':^6}")
    print("="*85)

    best_mcc = -1
    patience = 0
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # --- Training (AMP Enabled) ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False, ncols=80)
        
        for seq, adj, label in pbar:
            seq, adj, label = seq.to(Config.DEVICE), adj.to(Config.DEVICE), label.to(Config.DEVICE)
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                logits = model(seq, adj)
                loss = criterion(logits, label.unsqueeze(1))
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for seq, adj, label in val_loader:
                seq, adj, label = seq.to(Config.DEVICE), adj.to(Config.DEVICE), label.to(Config.DEVICE)
                
                # 推理也可以用autocast加速
                with autocast():
                    logits = model(seq, adj)
                    v_loss = criterion(logits, label.unsqueeze(1))
                
                val_loss += v_loss.item()
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().float().numpy()) # 确保转回FP32存numpy
                val_targets.extend(label.cpu().float().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs).flatten()
        
        metrics = compute_metrics(val_targets, val_probs)
        elapsed = time.time() - start_time
        
        # --- Logging ---
        print(f"{epoch+1:^5} | {avg_train_loss:^8.4f} | {avg_val_loss:^8.4f} | {metrics['Accuracy']:^6.4f} | {metrics['F1']:^6.4f} | {metrics['MCC']:^6.4f} | {metrics['AUROC']:^6.4f} | {elapsed:^6.1f}s")
        
        new_row = {
            'Epoch': epoch + 1,
            'Train_Loss': avg_train_loss,
            'Val_Loss': avg_val_loss,
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1'],
            'MCC': metrics['MCC'],
            'AUROC': metrics['AUROC'],
            'AUPRC': metrics['AUPRC'],
            'Time': elapsed
        }
        log_df.loc[len(log_df)] = new_row
        log_df.to_csv(csv_path, index=False)
        
        # --- Checkpoint Saving ---
        # 你的要求：保存训练好的模型 (.pth) 和 预测数据 (.npz)
        if metrics['MCC'] > best_mcc:
            best_mcc = metrics['MCC']
            patience = 0
            
            # 1. 保存模型权重 (文件小，方便加载)
            torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, "best_model.pth"))
            
            # 2. 保存预测结果 (用于反复画图，无需推理)
            np.savez(os.path.join(Config.OUTPUT_DIR, "best_predictions.npz"), 
                     y_true=val_targets, y_pred=val_probs)
            
            print(f"      >>> New Best MCC: {best_mcc:.4f} (Saved .pth & .npz)")
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                print(f"\n[Early Stopping] Stopped at epoch {epoch+1}")
                break

    print("\n" + "="*85)
    print(f"Training Complete.")
    print(f"1. Model saved to: {os.path.join(Config.OUTPUT_DIR, 'best_model.pth')}")
    print(f"2. Plot data saved to: {os.path.join(Config.OUTPUT_DIR, 'best_predictions.npz')}")
    print(f"3. Logs saved to: {csv_path}")

if __name__ == "__main__":
    run_experiment()