import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

class Config:
    MAX_LEN = 80       
    VOCAB = {'<PAD>': 0, 'A': 1, 'C': 2, 'G': 3, 'U': 4, 'N': 5}
    EMBED_DIM = 64      
    HIDDEN_DIM = 128    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
def encode_sequence(seq):
    seq = str(seq)[:Config.MAX_LEN]
    indices = [Config.VOCAB.get(char, 5) for char in seq]
    if len(indices) < Config.MAX_LEN:
        indices += [0] * (Config.MAX_LEN - len(indices))
    return torch.tensor([indices], dtype=torch.long)

def dbn_to_adj(dbn):
    dbn = str(dbn)
    length = min(len(dbn), Config.MAX_LEN)
    adj = np.zeros((Config.MAX_LEN, Config.MAX_LEN), dtype=np.float32)
    
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
    return torch.tensor([adj], dtype=torch.float32)

def load_trained_model(model_path):
    print(f"{model_path} ...")
    model = RiboMotifNet(use_structure=True).to(Config.DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval() 
    return model

def predict_single(model, sequence, structure):
    
    seq_tensor = encode_sequence(sequence).to(Config.DEVICE)
    adj_tensor = dbn_to_adj(structure).to(Config.DEVICE)
    
    with torch.no_grad():
        logits = model(seq_tensor, adj_tensor)
        probability = torch.sigmoid(logits).item()
        
    prediction = 1 if probability > 0.5 else 0
    return probability, prediction

def predict_batch_from_file(model, input_file, output_file):
    
    print(f"{input_file} ...")
    sep = '\t' if input_file.endswith('.tsv') else ','
    df = pd.read_csv(input_file, sep=sep)
    
    if 'sequence' not in df.columns or 'structure' not in df.columns:
        raise ValueError("Must Contain'sequence' 和 'structure' ")
    
    probs = []
    preds = []
    
    print("batch predict...")
    for index, row in df.iterrows():
        prob, pred = predict_single(model, row['sequence'], row['structure'])
        probs.append(prob)
        preds.append(pred)
        
        if (index + 1) % 1000 == 0:
            print(f"Processsed {index + 1} / {len(df)} ...")
            
    df['predicted_prob'] = probs
    df['predicted_label'] = preds
    
    # 保存结果
    df.to_csv(output_file, sep=sep, index=False)
    print(f"Prediction completed,saved to{output_file}")


if __name__ == "__main__":
    MODEL_PATH = "training_logs/best_model.pth" 
    
    model = load_trained_model(MODEL_PATH)
    print(f"{Config.DEVICE}")
    print("-" * 50)
    test_seq = "AUGGCCAUGGCGCCA"
    test_struct = ".((....))......" 
    prob, pred = predict_single(model, test_seq, test_struct)
    print(f"{test_seq}")
    print(f"{test_struct}")
    print(f"{prob:.4f}")
    print(f"{'Functional sequence' if pred == 1 else 'No functional sequence'}")
    print("-" * 50)
    
    ###modelB：batch predict
    ###Input new_data.tsv for predict
    # INPUT_FILE = "new_data.tsv"
    # OUTPUT_FILE = "predictions_result.tsv"
    # predict_batch_from_file(model, INPUT_FILE, OUTPUT_FILE)