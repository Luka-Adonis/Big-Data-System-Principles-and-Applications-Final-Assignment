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
import os

# ==========================================
# 1. 配置参数 (基于你的最优发现)
# ==========================================
FILE_PATH = '2024_Wimbledon_featured_matches.csv'
WINDOW_SIZE = 5       
HIDDEN_DIM = 64       
BATCH_SIZE = 32
EPOCHS = 100           
LEARNING_RATE = 0.0001
TEST_SIZE = 0.1       # 你发现的 9:1 比例

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

    return get_seq(train_m), get_seq(test_m), test_m

# ==========================================
# 3. 模型与可视化工具
# ==========================================
class MomentumLSTM(nn.Module):
    def __init__(self, in_dim):
        super(MomentumLSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, HIDDEN_DIM, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def plot_results(losses, y_true, y_probs):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Curve
    axes[0].plot(losses, color='#1f77b4', lw=2)
    axes[0].set_title('Training Loss Convergence')
    axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Loss')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title('ROC Curve'); axes[1].legend()

    # 3. Probability Distribution
    sns.histplot(y_probs, bins=20, kde=True, ax=axes[2], color='#2ca02c')
    axes[2].set_title('Prediction Confidence Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_momentum_heatmap(X_test, y_probs, feature_names):
    """
    计算并绘制特征与动量分值之间的相关性热力图
    """
    # 1. 提取序列中的最后一个时间步（当前点），因为 momentum score 是针对当前点预测的
    # X_test 维度是 [Batch, Window, Features]，我们取 [:, -1, :]
    X_current_point = X_test[:, -1, :].numpy() 
    
    # 2. 整合为 DataFrame
    df_analysis = pd.DataFrame(X_current_point, columns=feature_names)
    df_analysis['Predicted_Momentum'] = y_probs.flatten()
    
    # 3. 计算相关系数矩阵
    # 我们主要关心其他特征与 Predicted_Momentum 的关系
    corr_matrix = df_analysis.corr()
    momentum_corr = corr_matrix[['Predicted_Momentum']].sort_values(by='Predicted_Momentum', ascending=False)
    
    # 4. 绘图
    plt.figure(figsize=(8, 10))
    sns.set_theme(style="white")
    
    # 绘制热力图
    sns.heatmap(momentum_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
    
    plt.title('Feature Correlation with Momentum Score (LSTM)', fontsize=15)
    plt.ylabel('Input Features')
    plt.show()

    # 附加：打印简要的解释结论
    top_pos = momentum_corr.index[1] # 第一个是它自己
    top_neg = momentum_corr.index[-1]
    print(f"💡 解释性分析: 动量的最强正相关特征是 '{top_pos}'，最强负相关特征是 '{top_neg}'。")

# ==========================================
# 4. 主程序
# ==========================================
def main():
    X_raw, y_raw, m_ids, f_cols = load_and_preprocess(FILE_PATH)
    (X_train, y_train), (X_test, y_test), test_match_list = prepare_sequences(X_raw, y_raw, m_ids)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MomentumLSTM(len(f_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    loss_history = []
    print(f"🚀 训练开始 | 模式: 9:1 划分 | 设备: {device}")

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

    # 评估与可视化
    model.eval()
    with torch.no_grad():
        y_probs = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()
        y_preds = (y_probs > 0.5).astype(int)

    print("\n" + "="*40)
    print(f"📊 性能评估报告:")
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