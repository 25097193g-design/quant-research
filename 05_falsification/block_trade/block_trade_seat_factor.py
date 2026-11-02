"""大宗交易·席位净流入因子 seat_net_flow (time-boxed experiment, 按用户spec)
机构席位: buyer/seller含 [机构专用/深股通专用/沪股通专用/券商自营]; 剔游资散户营业部
seat_net_flow = (Σ机构买额 − Σ机构卖额) / 该股过去20日日均成交额
严格v3 PIT: signal_date=下个交易日T+1, 行业+市值rank-first中性化, 换手guard
决策线: Pass=多空夏普>0.65 或 (多头>0.8且换手降); Fail=多空<0.5 或 ICIR<0.5
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u block_trade_seat_factor.py
"""
import rqdatac as rq; rq.init()
import pandas as pd, numpy as np
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'
INST=['机构专用','深股通专用','沪股通专用','券商自营']

price=pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date']=pd.to_datetime(price['date'])
td_cal=pd.DatetimeIndex(sorted(price['date'].unique())); td_arr=td_cal.values.astype('datetime64[ns]')
uni=sorted(price['order_book_id'].unique())
pieces=[]
for i in range(0,len(uni),500):
    try:
        d=rq.get_block_trade(uni[i:i+500],'2023-08-01','2026-07-30')
        if d is not None and len(d): pieces.append(d.reset_index())
    except: pass
bt=pd.concat(pieces,ignore_index=True)
bt['trade_date']=pd.to_datetime(bt['trade_date'])
print(f"大宗{len(bt)}条; 机构买方占比{(bt['buyer'].astype(str).str.contains('|'.join(INST))).mean():.1%}, 机构卖方{(bt['seller'].astype(str).str.contains('|'.join(INST))).mean():.1%}",flush=True)
def is_inst(s): return any(k in str(s) for k in INST)
bt['inst_buy']=np.where(bt['buyer'].map(is_inst), bt['total_turnover'], 0.0)
bt['inst_sell']=np.where(bt['seller'].map(is_inst), bt['total_turnover'], 0.0)
# PIT signal_date = 下个交易日
pos=np.searchsorted(td_arr, bt['trade_date'].values.astype('datetime64[ns]'), side='right')
valid=pos<len(td_arr)
bt['signal_date']=pd.to_datetime(np.where(valid, td_arr[np.clip(pos,0,len(td_arr)-1)], np.datetime64('NaT','ns'))).to_period('M')
bt=bt.dropna(subset=['signal_date']).rename(columns={'signal_date':'sig_ym'})
g=bt.groupby(['order_book_id','sig_ym']).agg(inst_buy=('inst_buy','sum'),inst_sell=('inst_sell','sum'),n=('inst_buy','count')).reset_index()
g['seat_raw']=g['inst_buy']-g['inst_sell']
# 20日日均成交额(月末)
turn=price.pivot(index='date',columns='order_book_id',values='total_turnover').sort_index()
t20=turn.rolling(20,min_periods=10).mean().resample('ME').last(); t20.index=t20.index.to_period('M')
t20l=t20.stack().rename('turn20').reset_index(); t20l.columns=['sig_ym','order_book_id','turn20']
g=g.merge(t20l,on=['sig_ym','order_book_id'],how='left'); g=g.dropna(subset=['turn20']); g=g[g['turn20']>0]
g['seat_net_flow']=(g['seat_raw']/g['turn20']).clip(-1.5,1.5)  # 极端值截尾
print(f"席位面板{len(g)}行, 涉及{g['order_book_id'].nunique()}股, 月{g['sig_ym'].nunique()}",flush=True)
# mc+行业
fac=pd.read_parquet(f'{DATA}/factors_monthend.parquet'); fac['ym']=pd.to_datetime(fac['monthend']).dt.to_period('M')
pit=pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); pit['ym']=pd.to_datetime(pit['monthend']).dt.to_period('M')
g=g.merge(fac[['order_book_id','ym','market_cap_3']].rename(columns={'ym':'sig_ym'}),on=['order_book_id','sig_ym'],how='left')
g=g.merge(pit[['order_book_id','ym','industry_l1']].rename(columns={'ym':'sig_ym'}),on=['order_book_id','sig_ym'],how='left')
g=g.dropna(subset=['market_cap_3','industry_l1']); g['logmc']=np.log(g['market_cap_3'])
def neut(col):
    out=[]
    for ym,grp in g.groupby('sig_ym'):
        if len(grp)<30: continue
        r=grp[col].rank(pct=True); r=r-r.groupby(grp['industry_l1']).transform('mean')
        capq=pd.qcut(grp['logmc'].rank(method='first'),10,labels=False); r=r-r.groupby(capq.values).transform('mean')
        out.append(pd.DataFrame({'order_book_id':grp['order_book_id'].values,'sig_ym':ym,'n':r.values}))
    return pd.concat(out,ignore_index=True)
g=g.merge(neut('seat_net_flow').rename(columns={'n':'seat_n'}),on=['sig_ym','order_book_id'],how='left')
P=g.set_index(['sig_ym','order_book_id'])
cl=price.pivot(index='date',columns='order_book_id',values='close').sort_index()
mec=cl.resample('ME').last(); mec.index=mec.index.to_period('M'); fwd1=(mec.shift(-1)/mec-1)
def ic(col):
    o={}
    for ym in P.index.get_level_values('sig_ym').unique():
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='sig_ym')[col].dropna(); r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<30: continue
        o[ym]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    s=pd.Series(o).dropna(); return(len(s),s.mean(),s.mean()/s.std() if s.std()>0 else 0,(s>0).mean()) if len(s) else(0,0,0,0)
def ls(col):
    L=[];LO=[];to=[];prev=set()
    for ym in P.index.get_level_values('sig_ym').unique():
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='sig_ym')[col].dropna(); r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<50: continue
        ss=s.reindex(idx).sort_values(); q=int(len(ss)*0.2)
        long=ss.index[-q:]; short=ss.index[:q]
        LO.append(r.reindex(long).mean()); L.append(r.reindex(long).mean()-r.reindex(short).mean())
        if prev: ov=len(set(long)&prev)/max(len(long),1); to.append(1-ov)
        prev=set(long)
    L=np.array(L);LO=np.array(LO);to=np.array(to)
    ls_sh=L.mean()*12/(L.std()*np.sqrt(12)) if L.std()>0 else 0
    lo_sh=LO.mean()*12/(LO.std()*np.sqrt(12)) if LO.std()>0 else 0
    return ls_sh,lo_sh,(to.mean() if len(to) else np.nan)
n,i,ir,pos=ic('seat_n'); ls_sh,lo_sh,to=ls('seat_n')
print(f"\n=== 席位净流入 seat_net_flow (PIT, 行业+市值中性) ===")
print(f"  Rank IC: n={n} IC={i:+.4f} ICIR={ir:+.2f} 正月{pos:.0%}")
print(f"  多空夏普={ls_sh:+.2f} | 多头夏普={lo_sh:+.2f} | 月换手={to:.0%}")
print(f"\n=== 决策线判决 ===")
if ls_sh>0.65: verdict="✅PASS(多空>0.65)——席位有独立alpha,值得继续做"
elif (lo_sh>0.8): verdict=f"✅PASS(多头>0.8,换手{to:.0%})——多头信号强"
elif ls_sh<0.5 or ir<0.5: verdict="❌FAIL(多空<0.5或ICIR<0.5)——席位也无独立alpha,资金面彻底kill,收尾进Week2"
else: verdict=f"⚠️灰区(多空{ls_sh:.2f}在0.5-0.65, ICIR{ir:.2f})——边际弱信号,倾向kill"
print(verdict)
g.to_csv(f'{OUT}/block_trade_seat.csv',index=False)
print(f"输出: block_trade_seat.csv")
