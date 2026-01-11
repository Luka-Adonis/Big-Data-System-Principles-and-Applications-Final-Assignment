import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, roc_curve, auc, mean_absolute_error, mean_squared_error)
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==========================================
# 1. 配置参数 (针对小样本 Transformer 进行优化)
# ==========================================
FILE_PATH = '2024_Wimbledon_featured_matches.csv'
WINDOW_SIZE = 5       
D_MODEL = 32          # 减小维度，降低过拟合风险
NHEAD = 2             # 减少头数
NUM_LAYERS = 1        # 减少层数，使模型更易训练
BATCH_SIZE = 32
EPOCHS = 200          # Transformer 收敛慢，增加轮数
LEARNING_RATE = 0.0001 # 降低学习率，防止梯度爆炸
TEST_SIZE = 0.1       

# ==========================================
# 2. 数据预处理
# ==========================================
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}
    df['p1_score_val'] = df['p1_score'].astype(str).map(score_map).fillna(0)
    df['p2_score_val'] = df['p2_score'].astype(str).map(score_map).fillna(0)
    
    for p in ['p1', 'p2']:
        dist_col = f'{p}_distance_run'
        m_mean = df.groupby('match_id')[dist_col].transform('mean')
        m_std = df.groupby('match_id')[dist_col].transform('std')
        df[f'{p}_dist_std'] = (df[dist_col] - m_mean) / (m_std.replace(0, 1e-6))
    
    df['dist_diff_std'] = df['p1_dist_std'] - df['p2_dist_std']
    df['is_p1_server'] = (df['server'] == 1).astype(int)

    feature_cols = [
        'p1_score_val', 'p2_score_val', 'p1_games', 'p2_games', 'p1_sets', 'p2_sets',
        'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault',
        'p1_unf_err', 'p2_unf_err', 'serve_no', 'dist_diff_std', 
        'is_p1_server', 'rally_count', 'speed_mph'
    ]
    
    X_raw = df[feature_cols].fillna(0).values
    y_raw = (df['point_victor'] == 2).astype(int).values 
    match_ids = df['match_id'].values
    return X_raw, y_raw, match_ids, feature_cols

def prepare_sequences(X, y, match_ids):
    unique_matches = np.unique(match_ids)
    train_m, test_m = train_test_split(unique_matches, test_size=TEST_SIZE, random_state=42)
    
    def get_seq(m_list):
        X_s, y_s = [], []
        for mid in m_list:
            mask = (match_ids == mid)
            m_X, m_y = X[mask], y[mask]
            if len(m_X) > WINDOW_SIZE:
                for i in range(WINDOW_SIZE, len(m_X)):
                    X_s.append(m_X[i-WINDOW_SIZE : i]) 
                    y_s.append(m_y[i]) 
        return torch.FloatTensor(np.array(X_s)), torch.FloatTensor(np.array(y_s)).view(-1, 1)

    return get_seq(train_m), get_seq(test_m)

# ==========================================
# 3. 优化后的 Transformer 模型
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MomentumTransformer(nn.Module):
    def __init__(self, in_dim):
        super(MomentumTransformer, self).__init__()
        self.embedding = nn.Linear(in_dim, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        
        # 使用 Pre-LN 结构提高训练稳定性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_MODEL*2, 
            dropout=0.5, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # 平均池化：综合整段序列的动量特征
        output = torch.mean(output, dim=1) 
        return self.fc(output)

# ==========================================
# 4. 可视化函数
# ==========================================
def plot_results(losses, y_true, y_probs):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(losses, color='#1f77b4', lw=2)
    axes[0].set_title('Training Loss Convergence')
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title('ROC Curve'); axes[1].legend()

    sns.histplot(y_probs, bins=20, kde=True, ax=axes[2], color='#2ca02c')
    axes[2].set_title('Prediction Confidence Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_momentum_heatmap(X_test, y_probs, feature_names):
    """
    生成特征与动量分值的相关性热力图
    """
    # 1. 将测试集数据平铺（取每个序列的最后一帧，即当前预测点）
    X_last_step = X_test[:, -1, :].numpy() 
    
    # 2. 构建 DataFrame
    df_heatmap = pd.DataFrame(X_last_step, columns=feature_names)
    df_heatmap['Momentum_Score'] = y_probs.flatten()
    
    # 3. 计算相关系数矩阵
    corr_matrix = df_heatmap.corr()
    
    # 4. 绘图
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    
    # 专门筛选出各个特征与 Momentum_Score 的相关性，并排序
    momentum_corr = corr_matrix[['Momentum_Score']].sort_values(by='Momentum_Score', ascending=False)
    
    sns.heatmap(momentum_corr, annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation with Predicted Momentum Score')
    plt.show()

    # (可选) 也可以画完整的相关性矩阵图
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # 遮住上半部分，美观
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".1f", square=True)
    plt.title('Global Feature Interaction Matrix')
    plt.show()

# ==========================================
# 5. 执行主逻辑
# ==========================================
def main():
    X_raw, y_raw, m_ids, f_cols = load_and_preprocess(FILE_PATH)
    (X_train, y_train), (X_test, y_test) = prepare_sequences(X_raw, y_raw, m_ids)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MomentumTransformer(len(f_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    loss_history = []
    print(f"🚀 Transformer 抢救训练开始 | 设备: {device}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss/len(train_loader))
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {loss_history[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        y_probs = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()
        y_preds = (y_probs > 0.5).astype(int)
        
    print("\n" + "="*40)
    print(f"📊 Transformer 完整性能评估报告:")
    print(f"Accuracy:  {accuracy_score(y_true, y_preds):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_preds):.4f}")
    print(f"AUC Score: {roc_auc_score(y_true, y_probs):.4f}")
    print(f"MAE:       {mean_absolute_error(y_true, y_probs):.4f}")
    print(f"RMSE:      {np.sqrt(mean_squared_error(y_true, y_probs)):.4f}")
    print("="*40)
    
    plot_results(loss_history, y_true, y_probs)
    plot_momentum_heatmap(X_test, y_probs, f_cols)

if __name__ == "__main__":
    main()