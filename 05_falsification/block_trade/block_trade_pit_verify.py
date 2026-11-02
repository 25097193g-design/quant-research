"""隔离验证: v2→v3的夏普崩跌(1.21→0.34)到底是PIT日期shift造成, 还是v3有bug
同一套代码, 只换 聚合月份=trade_ym(v2) vs sig_ym(v3 PIT), 其余(中性化rank法/前向收益/LS)完全一致
如果trade_ym还原~1.21、sig_ym~0.34 → 纯日期效应(信号fragile/前视帮极端值); 若trade_ym也~0.34 → v2的1.21是别的artifact
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u block_trade_pit_verify.py
"""
import rqdatac as rq; rq.init()
import pandas as pd, numpy as np
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'
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
bt['trade_date']=pd.to_datetime(bt['trade_date']); bt['trade_ym']=bt['trade_date'].dt.to_period('M')
pos=np.searchsorted(td_arr, bt['trade_date'].values.astype('datetime64[ns]'), side='right')
valid=pos<len(td_arr)
bt['sig_ym']=pd.to_datetime(np.where(valid, td_arr[np.clip(pos,0,len(td_arr)-1)], np.datetime64('NaT','ns'))).to_period('M')
cl=price[['order_book_id','date','close']]
bt=bt.merge(cl,left_on=['order_book_id','trade_date'],right_on=['order_book_id','date'],how='left',suffixes=('','_c'))
bt['prem']=(bt['price']/bt['close']-1)
fac=pd.read_parquet(f'{DATA}/factors_monthend.parquet'); fac['ym']=pd.to_datetime(fac['monthend']).dt.to_period('M')
pit=pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); pit['ym']=pd.to_datetime(pit['monthend']).dt.to_period('M')
mec=price.pivot(index='date',columns='order_book_id',values='close').sort_index().resample('ME').last(); mec.index=mec.index.to_period('M')
fwd1=(mec.shift(-1)/mec-1)

def run(ymcol,label):
    b=bt.dropna(subset=[ymcol]).copy()
    g=b.groupby(['order_book_id',ymcol]).agg(prem=('prem','mean'),btot=('total_turnover','sum')).reset_index().rename(columns={ymcol:'ym'})
    g=g.merge(fac[['order_book_id','ym','market_cap_3']],on=['order_book_id','ym'],how='left').merge(pit[['order_book_id','ym','industry_l1']],on=['order_book_id','ym'],how='left')
    g=g.dropna(subset=['market_cap_3','industry_l1']); g['logmc']=np.log(g['market_cap_3'])
    # neut (rank method first, 同v3)
    def neut(col):
        out=[]
        for ym,grp in g.groupby('ym'):
            if len(grp)<30: continue
            r=grp[col].rank(pct=True); r=r-r.groupby(grp['industry_l1']).transform('mean')
            capq=pd.qcut(grp['logmc'].rank(method='first'),10,labels=False); r=r-r.groupby(capq.values).transform('mean')
            out.append(pd.DataFrame({'order_book_id':grp['order_book_id'].values,'ym':ym,'n':r.values}))
        return pd.concat(out,ignore_index=True)
    g=g.merge(neut('prem').rename(columns={'n':'prem_n'}),on=['ym','order_book_id'],how='left')
    g=g.merge(neut('btot').rename(columns={'n':'btot_n'}),on=['ym','order_book_id'],how='left')
    g['comp']=g['prem_n']+0.5*g['btot_n']
    P=g.set_index(['ym','order_book_id'])
    # IC
    def ic(col):
        o={}
        for ym in P.index.get_level_values('ym').unique():
            if ym not in fwd1.index: continue
            s=P.xs(ym,level='ym')[col].dropna(); r=fwd1.loc[ym].reindex(s.index).dropna()
            idx=s.index.intersection(r.index)
            if len(idx)<30: continue
            o[ym]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
        s=pd.Series(o).dropna(); return (s.mean(), s.mean()/s.std() if s.std()>0 else 0) if len(s) else (0,0)
    # LS
    ls=[];lo=[]
    for ym in P.index.get_level_values('ym').unique():
        if ym not in fwd1.index: continue
        s=P.xs(ym,level='ym')['comp'].dropna(); r=fwd1.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<50: continue
        ss=s.reindex(idx).sort_values(); q=int(len(ss)*0.2)
        long=ss.index[-q:]; short=ss.index[:q]
        lo.append(r.reindex(long).mean()); ls.append(r.reindex(long).mean()-r.reindex(short).mean())
    ls=np.array(ls);lo=np.array(lo)
    pm_ic,pm_ir=ic('prem_n'); cp_ic,cp_ir=ic('comp')
    ls_sh=ls.mean()*12/(ls.std()*np.sqrt(12)) if ls.std()>0 else 0
    lo_sh=lo.mean()*12/(lo.std()*np.sqrt(12)) if lo.std()>0 else 0
    print(f"[{label}] prem ICIR={pm_ir:+.2f} | 复合IC={cp_ic:+.4f}/ICIR={cp_ir:+.2f} | 多空夏普={ls_sh:+.2f} 多头夏普={lo_sh:+.2f} (n月={len(ls)})")
run('trade_ym','v2 trade_date(前视)')
run('sig_ym','v3 PIT(T+1)')
print("\n读法: 若两者夏普差大→纯日期效应(信号fragile); 若都低→v2的1.21另有artifact")
