# P4c-2: 深NN —— per-stock标签归一 + 加宽结构 + 4种子
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 12_nn_deep.py
# 三处升级 vs 11_nn.py(0.2780):
#   1) 标签 = log target - enc_stock(个股过去label扩张均值, 只用过去, 部署合法)
#      → NN不用学6.7倍异质性, 只学残差(两阶段教训的直接应用)
#   2) 结构: LayerNorm + emb16 + 256-128-64(带dropout), AdamW + 余弦退火, 80轮
#   3) 4种子logit平均
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import rmspe

df = pd.read_parquet('cache/features_v3.parquet')
feat_cols = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]
u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr]

med = tr[feat_cols].median()
Xtr = tr[feat_cols].fillna(med).values.astype('float32')
Xva = va[feat_cols].fillna(med).values.astype('float32')
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
Str = tr['stock_id'].values.astype('int64')
Sva = va['stock_id'].values.astype('int64')
ytr = (np.log(tr['target'].values) - tr['enc_stock'].values).astype('float32')  # 残差标签
enc_va = va['enc_stock'].values.astype('float32')
tgt = va['target'].values

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device={dev} | 标签=残差(log target − enc_stock) | train {len(tr):,}')

class Net(nn.Module):
    def __init__(self, n, n_stock, emb=16):
        super().__init__()
        self.emb = nn.Embedding(n_stock, emb)
        self.ln = nn.LayerNorm(n)
        self.net = nn.Sequential(
            nn.Linear(n + emb, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, s):
        return self.net(torch.cat([self.ln(x), self.emb(s)], 1)).squeeze(-1)

def train(seed, epochs=80, bs=8192, patience=8):
    torch.manual_seed(seed)
    m = Net(len(feat_cols), int(Str.max()) + 1).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    Xt = torch.from_numpy(Xtr).to(dev)
    St = torch.from_numpy(Str).to(dev)
    Yt = torch.from_numpy(ytr).to(dev)
    Xv, Sv = torch.from_numpy(Xva).to(dev), torch.from_numpy(Sva).to(dev)
    best, bad = (9.0, None), 0
    n = len(Xt)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = nn.functional.mse_loss(m(Xt[idx], St[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            lg = np.clip(m(Xv, Sv).cpu().numpy(), -20, 5)
        s = rmspe(np.exp(lg + enc_va), tgt)
        if s < best[0]:
            best, bad = (s, lg.copy()), 0
        else:
            bad += 1
        if bad >= patience:
            break
    print(f'  seed={seed}: best RMSPE={best[0]:.4f} @epoch{ep}')
    return best[1]

preds = [train(s) for s in (42, 2024, 7, 99)]
p_nn = np.exp(np.mean(preds, axis=0) + enc_va)
print(f'深NN 4种子: RMSPE = {rmspe(p_nn, tgt):.4f}   [对照: 首版NN 0.2780 | B1 0.2493 | blend 0.2458]')

nn_out = va[['stock_id', 'time_id']].copy()
nn_out['p_nn2'] = p_nn
lg_ = pd.read_parquet('cache/twostage_val_pred.parquet')[['stock_id', 'time_id', 'p_b1', 'p_ts']]
mg = nn_out.merge(lg_, on=['stock_id', 'time_id']).merge(
    va[['stock_id', 'time_id', 'target']], on=['stock_id', 'time_id'])
t = mg['target'].values
w_nn, w_b1 = np.log(mg.p_nn2.values), np.log(mg.p_b1.values)
w_ts = np.log(mg.p_ts.values)
for name, combo in [('NN2+B1', (w_nn + w_b1) / 2),
                    ('NN2+B1+两阶段', (w_nn + w_b1 + w_ts) / 3),
                    ('NN2×2+B1+两阶段', (2 * w_nn + w_b1 + w_ts) / 4)]:
    print(f'blend({name}) RMSPE = {rmspe(np.exp(combo), t):.4f}')
mg.to_parquet('cache/nn2_val_pred.parquet')
ps = mg.assign(e=(mg.p_nn2.values - mg.target.values) / mg.target.values).groupby('stock_id').apply(
    lambda g: rmspe(g.p_nn2.values, g.target.values))
w3 = ps.sort_values(ascending=False).head(3)
print('深NN分股最差3只: ' + ', '.join(f'stock_{int(k)}={v:.3f}' for k, v in w3.items())
      + (f' | stock_31={ps.loc[31]:.3f}' if 31 in ps.index else ''))
