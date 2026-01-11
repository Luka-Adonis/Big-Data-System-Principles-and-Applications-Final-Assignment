import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools
import os

# ==========================================
# 1. 数据预处理函数
# ==========================================
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 比分映射 (打球前的状态)
    score_map = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}
    df['p1_score_val'] = df['p1_score'].astype(str).map(score_map).fillna(0)
    df['p2_score_val'] = df['p2_score'].astype(str).map(score_map).fillna(0)
    
    # 归一化跑动距离差
    for p in ['p1', 'p2']:
        dist_col = f'{p}_distance_run'
        m_mean = df.groupby('match_id')[dist_col].transform('mean')
        m_std = df.groupby('match_id')[dist_col].transform('std')
        df[f'{p}_dist_std'] = (df[dist_col] - m_mean) / (m_std.replace(0, 1e-6))
    
    df['dist_diff_std'] = df['p1_dist_std'] - df['p2_dist_std']
    df['is_p1_server'] = (df['server'] == 1).astype(int)

    # 这里的特征在当前行 i 预测时，只取 i-WINDOW 到 i-1 行
    feature_cols = [
        'p1_score_val', 'p2_score_val', 'p1_games', 'p2_games', 'p1_sets', 'p2_sets',
        'p1_winner', 'p2_winner', 'p1_double_fault', 'p2_double_fault',
        'p1_unf_err', 'p2_unf_err', 'serve_no', 'dist_diff_std', 
        'is_p1_server', 'rally_count', 'speed_mph'
    ]
    
    X_raw = df[feature_cols].fillna(0).values
    y_raw = (df['point_victor'] == 2).astype(int).values # 2映射为1，1映射为0
    match_ids = df['match_id'].values
    
    return X_raw, y_raw, match_ids, feature_cols

# ==========================================
# 2. 动态模型定义
# ==========================================
class MomentumLSTM(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(MomentumLSTM, self).__init__()
        self.lstm = nn.LSTM(in_dim, h_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, in_dim]
        _, (hn, _) = self.lstm(x)
        # hn[-1] 是最后一层 LSTM 的输出
        return self.fc(hn[-1])

# ==========================================
# 3. 单次实验运行函数
# ==========================================
def run_experiment(X_raw, y_raw, m_ids, f_cols, config):
    # 根据当前 Window Size 准备序列
    def get_seq(m_list, win_size):
        X_s, y_s = [], []
        for mid in m_list:
            mask = (m_ids == mid)
            m_X, m_y = X_raw[mask], y_raw[mask]
            if len(m_X) > win_size:
                for i in range(win_size, len(m_X)):
                    # 取 i-win 到 i-1 行作为输入 (历史)
                    # 预测第 i 行的结果 (当下)
                    X_s.append(m_X[i-win_size : i])
                    y_s.append(m_y[i])
        return torch.FloatTensor(np.array(X_s)), torch.FloatTensor(np.array(y_s)).view(-1, 1)

    unique_matches = np.unique(m_ids)
    train_m, test_m = train_test_split(unique_matches, test_size=0.1, random_state=42)
    
    X_train, y_train = get_seq(train_m, config['win'])
    X_test, y_test = get_seq(test_m, config['win'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    model = MomentumLSTM(len(f_cols), config['hidden']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = nn.BCELoss()

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    # 评估
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()
        auc = roc_auc_score(y_true, test_probs)
        acc = accuracy_score(y_true, (test_probs > 0.5).astype(int))
    
    return auc, acc

# ==========================================
# 4. 网格搜索主逻辑
# ==========================================
def main():
    file_path = '2024_Wimbledon_featured_matches.csv'
    X_raw, y_raw, m_ids, f_cols = load_and_preprocess(file_path)

    # --- 超参数搜索空间 (可根据需要修改) ---
    search_space = {
        'win': [5, 8, 12],            # 序列长度
        'hidden': [64, 128],          # LSTM单元数
        'epochs': [30, 50, 100],           # 训练轮数
        'lr': [0.001, 0.0005, 0.0001] # 学习率
    }

    # 生成组合
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔍 开始超参数调优，共 {len(combinations)} 组配置...")
    print(f"{'Win':<4} | {'Hid':<4} | {'Ep':<3} | {'LR':<7} | {'AUC':<7} | {'ACC':<7}")
    print("-" * 55)

    results = []
    best_auc = 0
    best_config = None

    for config in combinations:
        try:
            auc, acc = run_experiment(X_raw, y_raw, m_ids, f_cols, config)
            results.append({**config, 'auc': auc, 'acc': acc})
            
            print(f"{config['win']:<4} | {config['hidden']:<4} | {config['epochs']:<3} | {config['lr']:<7} | {auc:.4f}  | {acc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_config = config
        except Exception as e:
            print(f"配置 {config} 运行出错: {e}")

    print("-" * 55)
    print("🏆 调优完成!")
    print(f"最佳配置: {best_config}")
    print(f"最高 AUC: {best_auc:.4f}")

    # 保存结果方便后续分析
    results_df = pd.DataFrame(results)
    results_df.to_csv('tuning_results.csv', index=False)
    print("📋 所有尝试结果已保存至 'tuning_results.csv'")

if __name__ == "__main__":

    main()
