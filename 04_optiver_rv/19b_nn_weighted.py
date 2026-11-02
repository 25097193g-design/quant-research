# 弹药1: 深NN换1/y²精确对齐loss(比值空间) + 全家族最终blend对齐
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 19b_nn_weighted.py
# 设计(精确对齐RMSPE): est=exp(enc_stock)个股水平估计, r=target/est(比值);
#   (ŷ−y)/y ≡ (r̂−r)/r (est约掉) → 标准化z=(r−m)/s上做加权MSE, 权重1/r² = 精确最小化Σ(Δr)²/r²
# 输出: r̂=ẑ·s+m(clip), ŷ=est·r̂; 结构同12(LN+emb16+256-128-64+AdamW余弦+4种子)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import rmspe

df = pd.read_parquet('cache/features_v3.parquet')
feat_cols = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]
u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr].sort_values(
    ['stock_id', 'time_id']).reset_index(drop=True)

med = tr[feat_cols].median()
Xtr = tr[feat_cols].fillna(med).values.astype('float32')
Xva = va[feat_cols].fillna(med).values.astype('float32')
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
Str = tr['stock_id'].values.astype('int64')
Sva = va['stock_id'].values.astype('int64')

est_tr = np.exp(tr['enc_stock'].values).astype('float32')
est_va = np.exp(va['enc_stock'].values).astype('float32')
r_tr = (tr['target'].values / est_tr).astype('float32')
m_r, s_r = r_tr.mean(), r_tr.std()
ztr = ((r_tr - m_r) / s_r).astype('float32')
w_tr = (1.0 / r_tr ** 2).astype('float32')
w_tr = (w_tr / w_tr.mean()).astype('float32')  # 权重归一到均值1, 稳定lr
tgt = va['target'].values

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device={dev} | 比值r: mean={m_r:.3f} std={s_r:.3f} | train {len(tr):,}')

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
    m_ = Net(len(feat_cols), int(Str.max()) + 1).to(dev)
    opt = torch.optim.AdamW(m_.parameters(), lr=1.5e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    Xt = torch.from_numpy(Xtr).to(dev)
    St = torch.from_numpy(Str).to(dev)
    Zt = torch.from_numpy(ztr).to(dev)
    Wt = torch.from_numpy(w_tr).to(dev)
    Xv, Sv = torch.from_numpy(Xva).to(dev), torch.from_numpy(Sva).to(dev)
    best, bad = (9.0, None), 0
    n = len(Xt)
    for ep in range(epochs):
        m_.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = (Wt[idx] * (m_(Xt[idx], St[idx]) - Zt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        m_.eval()
        with torch.no_grad():
            zh = m_(Xv, Sv).cpu().numpy()
        r_hat = np.clip(zh * s_r + m_r, 1e-3, None)
        s = rmspe(est_va * r_hat, tgt)
        if s < best[0]:
            best, bad = (s, zh.copy()), 0
        else:
            bad += 1
        if bad >= patience:
            break
    print(f'  seed={seed}: best RMSPE={best[0]:.4f} @epoch{ep}')
    return best[1]

zh_all = [train(s) for s in (42, 2024, 7, 99)]
z_mean = np.mean(zh_all, axis=0)
r_hat = np.clip(z_mean * s_r + m_r, 1e-3, None)
p_nn3 = est_va * r_hat
print(f'NN-W 4种子: RMSPE = {rmspe(p_nn3, tgt):.4f}  [log残差版 0.2470 | HIS 0.2242]')

# ---- 全家族最终对齐(全部键合并) ----
nout = va[['stock_id', 'time_id', 'target']].copy()
nout['p_nn3'] = p_nn3
h1000 = pd.read_parquet('cache/his_val_pred.parquet')[['stock_id', 'time_id', 'p_his']]
s23 = pd.read_parquet('cache/shots23_val_pred.parquet')[['stock_id', 'time_id', 'p_his3', 'p_b2']]
mg = nout.merge(h1000, on=['stock_id', 'time_id']).merge(s23, on=['stock_id', 'time_id'])
t = mg['target'].values
L = lambda c: np.log(np.clip(mg[c].values, 1e-12, None))
combos = {
    'HIS1000+B2': (L('p_his') + L('p_b2')) / 2,
    'HIS1000+HIS3+B2': (L('p_his') + L('p_his3') + L('p_b2')) / 3,
    'HIS1000+B2+NNW': (L('p_his') + L('p_b2') + L('p_nn3')) / 3,
    'HIS1000×2+B2+NNW': (2 * L('p_his') + L('p_b2') + L('p_nn3')) / 4,
    'HIS1000+HIS3+B2+NNW': (L('p_his') + L('p_his3') + L('p_b2') + L('p_nn3')) / 4,
}
print('\n[最终对齐] (我们的split):')
for k, c in combos.items():
    print(f'  {k:<24} RMSPE = {rmspe(np.exp(c), t):.4f}')
mg.to_parquet('cache/final_family_pred.parquet')
