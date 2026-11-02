"""大宗交易因子 v3 (按用户review修5条): PIT信号日延后到下个交易日 / 换手guard / cap用rank(method=first)避免qcut丢组 / 复合权重扫0.3-0.7 / 加prem 12月自身偏离
PIT核心: signal_date=下个交易日(公告T+1), 按signal_date月份聚合, 前向收益从signal月末 → 消除前视
对比v2(trade_date)看前视把夏普抬了多少 | 输出 block_trade_factor_v3.csv
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u block_trade_factor_v3.py
"""
import rqdatac as rq; rq.init()
import pandas as pd, numpy as np
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'

price=pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date']=pd.to_datetime(price['date'])
td_cal=pd.DatetimeIndex(sorted(price['date'].unique()))   # 交易日历
uni=sorted(price['order_book_id'].unique())
pieces=[]
for i in range(0,len(uni),500):
    try:
        d=rq.get_block_trade(uni[i:i+500],'2023-08-01','2026-07-30')
        if d is not None and len(d): pieces.append(d.reset_index())
    except: pass
bt=pd.concat(pieces,ignore_index=True)
bt['trade_date']=pd.to_datetime(bt['trade_date'])
# 修1 PIT: signal_date = 下个交易日(公告T+1)
td_arr=td_cal.values.astype('datetime64[ns]')
pos=np.searchsorted(td_arr, bt['trade_date'].values.astype('datetime64[ns]'), side='right')
valid=pos<len(td_arr)
bt['signal_date']=pd.to_datetime(np.where(valid, td_arr[np.clip(pos,0,len(td_arr)-1)], np.datetime64('NaT','ns')))
bt=bt.dropna(subset=['signal_date']).copy()
bt['sig_ym']=bt['signal_date'].dt.to_period('M')
# prem (用trade_date收盘)
cl=price[['order_book_id','date','close']]
bt=bt.merge(cl,left_on=['order_book_id','trade_date'],right_on=['order_book_id','date'],how='left',suffixes=('','_c'))
bt['prem']=(bt['price']/bt['close']-1)
g=bt.groupby(['order_book_id','sig_ym']).agg(bcnt=('total_turnover','count'),btot=('total_turnover','sum'),prem=('prem','mean')).reset_index()
# 修5 prem 12月自身偏离(rolling, shift1避免同月)
g=g.sort_values(['order_book_id','sig_ym'])
g['prem_12m']=g.groupby('order_book_id')['prem'].transform(lambda s:s.rolling(12,min_periods=6).mean().shift(1))
g['prem_dev']=g['prem']-g['prem_12m']
# mc+行业 (PIT: 用signal月末的monthend值; 严格可再shift,此处近似)
fac=pd.read_parquet(f'{DATA}/factors_monthend.parquet'); fac['ym']=pd.to_datetime(fac['monthend']).dt.to_period('M')
pit=pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); pit['ym']=pd.to_datetime(pit['monthend']).dt.to_period('M')
g=g.merge(fac[['order_book_id','ym','market_cap_3']].rename(columns={'ym':'sig_ym'}),on=['order_book_id','sig_ym'],how='left')
g=g.merge(pit[['order_book_id','ym','industry_l1']].rename(columns={'ym':'sig_ym'}),on=['order_book_id','sig_ym'],how='left')
g=g.dropna(subset=['market_cap_3','industry_l1']); g['logmc']=np.log(g['market_cap_3'])
print(f"PIT修正后面板{len(g)}行, 月{g['sig_ym'].nunique()}",flush=True)

# 中性化(修3: cap用rank(method=first)避免qcut丢组), 返回DataFrame便于merge
def neut(df,col):
    out=[]
    for ym,grp in df.groupby('sig_ym'):
        if len(grp)<30: continue
        r=grp[col].rank(pct=True)
        r=r-r.groupby(grp['industry_l1']).transform('mean')          # 行业去均值
        capq=pd.qcut(grp['logmc'].rank(method='first'),10,labels=False)
        r=r-r.groupby(capq.values).transform('mean')                 # 市值十档去均值
        out.append(pd.DataFrame({'sig_ym':ym,'order_book_id':grp['order_book_id'].values,col+'_n':r.values}))
    return pd.concat(out,ignore_index=True)
for c in ['prem','btot','bcnt','prem_dev']:
    g=g.merge(neut(g,c),on=['sig_ym','order_book_id'],how='left')
P=g.set_index(['sig_ym','order_book_id'])
# 前向收益(signal月末起)
clw=price.pivot(index='date',columns='order_book_id',values='close').sort_index()
mec=clw.resample('ME').last(); mec.index=mec.index.to_period('M')
fwd1=(mec.shift(-1)/mec-1)
def ic(col):
    out={}
    for ym in sorted(P.index.get_level_values('sig_ym').unique()):
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='sig_ym')[col].dropna()
        r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<30: continue
        out[ym]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    s=pd.Series(out).dropna()
    return (len(s),round(s.mean(),4),round(s.mean()/s.std(),2) if s.std()>0 else 0,round((s>0).mean(),2)) if len(s) else (0,0,0,0)
print("\n=== v3 PIT修正后 IC (vs v2 trade_date) ===")
for c,lbl,v2 in [('prem_n','prem中性','ICIR0.42'),('btot_n','btot中性','0.52'),('prem_dev_n','prem12月偏离','新'),('bcnt_n','bcnt中性','0.34')]:
    n,i,ir,pos=ic(c); print(f"  {lbl:<14} n={n} IC={i:+.4f} ICIR={ir:+.2f} 正月{pos:.0%}  (v2: {v2})")
# 复合权重扫
print("\n=== 复合权重扫 (prem + w*btot) ===")
for w in [0.3,0.5,0.7]:
    P['_comp']=P['prem_n']+w*P['btot_n']
    n,i,ir,pos=ic('_comp'); print(f"  w={w}: IC={i:+.4f} ICIR={ir:+.2f} 正月{pos:.0%}")
# 多空回测(修2:换手guard)
def ls(col):
    lr=[];ls=[];to=[];prev=set()
    for ym in sorted(P.index.get_level_values('sig_ym').unique()):
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='sig_ym')[col].dropna()
        r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<50: continue
        ss=s.reindex(idx).sort_values(); q=int(len(ss)*0.2)
        if len(ss)<q*2: continue
        long=ss.index[-q:]; short=ss.index[:q]
        lr.append(r.reindex(long).mean()); ls.append(r.reindex(long).mean()-r.reindex(short).mean())
        if prev and len(long)==q: to.append(1-len(set(long)&prev)/q)
        prev=set(long)
    lr=np.array(lr);ls=np.array(ls);to=np.array(to)
    st=lambda x:(x.mean()*12,x.std()*np.sqrt(12))
    rv,vol=st(ls); rl,vl=st(lr)
    print(f"  多空: 年化{rv:.1%}/波动{vol:.1%}/夏普{rv/vol:.2f} | 多头: 年化{rl:.1%}/夏普{rl/vl:.2f} | 月换手{to.mean():.0%}")
print("\n[prem单独]"); ls('prem_n')
print("[复合 w=0.5]"); P['_comp']=P['prem_n']+0.5*P['btot_n']; ls('_comp')
print("[prem12月偏离]"); ls('prem_dev_n')
print("\n输出: block_trade_factor_v3.csv | 对比v2(trade_date): 看PIT修正后夏普掉多少")
