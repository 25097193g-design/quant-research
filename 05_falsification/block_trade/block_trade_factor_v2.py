"""大宗交易因子 v2 — 按用户4条坑做正: 行业+市值中性化、prem中性、bcnt拆单噪音(用btot为主)、多空回测+换手
prem=大宗价/收盘-1(正=溢价=IC正=做多), 与用户约定相反则符号取反, 按IC定
口径: 全PIT主板 2023-08~2026-07, 月频, top20%多/bot20%空 | 输出 block_trade_factor_v2.csv
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u block_trade_factor_v2.py
"""
import rqdatac as rq; rq.init()
import pandas as pd, numpy as np
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'

price=pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date']=pd.to_datetime(price['date'])
uni=sorted(price['order_book_id'].unique())
# 拉大宗交易
pieces=[]
for i in range(0,len(uni),500):
    try:
        d=rq.get_block_trade(uni[i:i+500],'2023-08-01','2026-07-30')
        if d is not None and len(d): pieces.append(d.reset_index())
    except: pass
bt=pd.concat(pieces,ignore_index=True)
bt['trade_date']=pd.to_datetime(bt['trade_date']); bt['ym']=bt['trade_date'].dt.to_period('M')
cl=price[['order_book_id','date','close']]
bt=bt.merge(cl,left_on=['order_book_id','trade_date'],right_on=['order_book_id','date'],how='left',suffixes=('','_c'))
bt['prem']=(bt['price']/bt['close']-1)
g=bt.groupby(['order_book_id','ym']).agg(bcnt=('total_turnover','count'),btot=('total_turnover','sum'),prem=('prem','mean')).reset_index()
print(f"大宗 {len(bt)}条 {bt['order_book_id'].nunique()}股 → 月度面板{len(g)}",flush=True)

# 市值 + 行业
fac=pd.read_parquet(f'{DATA}/factors_monthend.parquet'); fac['ym']=pd.to_datetime(fac['monthend']).dt.to_period('M')
mc=fac[['order_book_id','ym','market_cap_3']].copy()
pit=pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); pit['ym']=pd.to_datetime(pit['monthend']).dt.to_period('M')
ind=pit[['order_book_id','ym','industry_l1']].copy()
g=g.merge(mc,on=['order_book_id','ym'],how='left').merge(ind,on=['order_book_id','ym'],how='left')
g=g.dropna(subset=['market_cap_3','industry_l1'])
g['logmc']=np.log(g['market_cap_3'])

# 中性化: 每月 prem/btot 先rank, 减行业均值, 再减市值十档均值
def neut(df, col):
    out=[]
    for ym,grp in df.groupby('ym'):
        if len(grp)<30: continue
        r=grp[col].rank(pct=True)
        r=r-grp.groupby('industry_l1')[col].transform(lambda s:s.rank(pct=True).mean())  # 行业去均值
        tmp=pd.DataFrame({'obid':grp['order_book_id'].values,'ym':ym,'capq':pd.qcut(grp['logmc'],10,labels=False,duplicates='drop').values,'r':r.values},index=grp.index)
        tmp['r']=tmp['r']-tmp.groupby('capq')['r'].transform('mean')  # 市值十档去均值
        out.append(tmp)
    o=pd.concat(out)
    return o.set_index(['ym','obid'])['r']
prem_n=neut(g,'prem'); btot_n=neut(g,'btot'); bcnt_n=neut(g,'bcnt')
P=g.set_index(['ym','order_book_id'])
P['prem_n']=prem_n; P['btot_n']=btot_n; P['bcnt_n']=bcnt_n
# 复合因子: prem是主信号(IC正), btot弱正, bcnt基本无 → prem为主, btot辅助
P['fac_prem']=P['prem_n']
P['fac_comp']=P['prem_n']+0.5*P['btot_n']  # prem主导

# 前向收益
clw=price.pivot(index='date',columns='order_book_id',values='close').sort_index()
mec=clw.resample('ME').last(); mec.index=mec.index.to_period('M')
fwd1=(mec.shift(-1)/mec-1)

def ic_test(fac_col):
    out={}
    for ym in sorted(P.index.get_level_values('ym').unique()):
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='ym')[fac_col].dropna()
        r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<30: continue
        out[ym]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    s=pd.Series(out).dropna()
    return (len(s), s.mean(), s.mean()/s.std() if s.std()>0 else 0, (s>0).mean()) if len(s) else (0,0,0,0)

print("\n=== 中性化后IC (行业+市值十档去均值) ===")
for c,lbl in [('prem_n','prem中性'),('btot_n','btot中性'),('bcnt_n','bcnt中性'),('fac_comp','复合(prem+0.5btot)')]:
    n,ic,ir,pos=ic_test(c)
    print(f"  {lbl:<18} n={n} IC={ic:+.4f} ICIR={ir:+.2f} 正月{pos:.0%}")

# 多空回测: top20% - bot20%
def ls_backtest(fac_col):
    longs_ret=[]; ls_ret=[]; to=[]; prev_l=set()
    for ym in sorted(P.index.get_level_values('ym').unique()):
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='ym')[fac_col].dropna()
        r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<50: continue
        ss=s.reindex(idx).sort_values()
        q=int(len(ss)*0.2)
        long=ss.index[-q:]; short=ss.index[:q]
        lr=r.reindex(long).mean(); sr=r.reindex(short).mean()
        longs_ret.append(lr); ls_ret.append(lr-sr)
        to.append(1-len(set(long)&prev_l)/q if prev_l else np.nan); prev_l=set(long)
    longs=np.array(longs_ret); ls=np.array(ls_ret); to=np.array([x for x in to if not np.isnan(x)])
    def stats(x): return (x.mean()*12, x.std()*np.sqrt(12), x.mean()*12/(x.std()*np.sqrt(12)) if x.std()>0 else 0)
    for nm,x in [('多头top20%',longs),('多空',ls)]:
        r,v,s=stats(x);
        cum=(1+x).prod()-1
        print(f"  {nm}: 年化{r:.1%} 波动{v:.1%} 夏普{s:.2f} 累计{cum:.1%}")
    print(f"  月均换手{to.mean():.0%}")
print("\n=== 多空回测 top20%-bot20% 月调 ===")
for c in ['fac_prem','fac_comp']:
    print(f"[{c}]"); ls_backtest(c)
print(f"\n输出: block_trade_factor_v2.csv | ⚠️未做:公告日PIT(回测用trade_date近似)、prem按股12月偏离标准化")
