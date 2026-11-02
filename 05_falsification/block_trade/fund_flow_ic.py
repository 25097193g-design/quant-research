"""主力净流入 kill-or-keep: 抽样200只, 日频截面IC(主力净流入净占比 today → 次日/5日收益)
akshare stock_individual_fund_flow(按股, 近120天) | daemon超时 | 输出 fund_flow_ic.csv
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u fund_flow_ic.py
"""
import akshare as ak, pandas as pd, numpy as np, time, threading, random
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'
random.seed(42)

# 抽样200只主板股(从mainboard_price列)
price=pd.read_parquet(f'{DATA}/mainboard_price.parquet')
allc=[c for c in price['order_book_id'].unique() if str(c).split('.')[0][0] in ('6','0') and not str(c).startswith('688')]
samp=random.sample(allc, 150)
print(f"抽样{len(samp)}只")

def call_timeout(fn,t=20):
    box=[None]
    def _r():
        try: box[0]=fn()
        except Exception: box[0]=None
    th=threading.Thread(target=_r,daemon=True); th.start(); th.join(t)
    return box[0]
def ob2ak(o):
    c=o.split('.')[0]; return (c,'sh' if c.startswith('6') else 'sz')

def fetch(code,mk):
    for _ in range(4):
        d=call_timeout(lambda c=code,m=mk: ak.stock_individual_fund_flow(c,m),t=25)
        if d is not None and len(d)>0: return d
        time.sleep(2.5)
    return None
pieces=[]; ok=0
for i,o in enumerate(samp):
    code,mk=ob2ak(o)
    d=fetch(code,mk)
    if d is None or len(d)==0:
        print(f"  {i+1}/{len(samp)} {o} 失败",flush=True); continue
    d=d.rename(columns={'日期':'date','主力净流入-净占比':'main_pct'})
    if 'main_pct' not in d.columns: continue
    d['date']=pd.to_datetime(d['date'])
    d=d[['date','main_pct']].assign(order_book_id=o)
    pieces.append(d); ok+=1
    if (i+1)%20==0: print(f"  {i+1}/{len(samp)} ok={ok}",flush=True)
    time.sleep(3.0)
ff=pd.concat(pieces,ignore_index=True).dropna()
ff=ff[ff['main_pct'].abs()<50]  # 去异常
print(f"主力净流入 panel: {ff.shape}, 股{ff['order_book_id'].nunique()}, 日{ff['date'].nunique()}({ff['date'].min().date()}~{ff['date'].max().date()})")

# 次日/5日收益 from mainboard_price
cl=price.assign(date=pd.to_datetime(price['date'])).pivot(index='date',columns='order_book_id',values='close').sort_index()
fwd1=cl.pct_change(1).shift(-1)  # 次日收益
fwd5=(cl.shift(-5)/cl-1)
def daily_ic(sig_wide, ret_wide):
    out={}
    for dt in sig_wide.index.intersection(ret_wide.index):
        s=sig_wide.loc[dt].dropna(); r=ret_wide.loc[dt].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<50: continue
        out[dt]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    return pd.Series(out)
sig=ff.pivot_table(index='date',columns='order_book_id',values='main_pct',aggfunc='last')
rows=[]
for name,ret in [('次日1d',fwd1),('未来5d',fwd5),('当日(对照)',cl.pct_change(1))]:
    s=daily_ic(sig,ret).dropna()
    if len(s)<10: continue
    rows.append({'信号':name,'n日':len(s),'IC均值':round(s.mean(),4),'ICIR':round(s.mean()/s.std(),2) if s.std()>0 else 0,'正日比':round((s>0).mean(),2)})
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/fund_flow_ic.csv',index=False)
print("\n=== 主力净流入 日频截面IC (抽样200只, 近120天) ===")
print(R.to_string(index=False))
print("\n读法: '当日'应强(主力买→当天涨,同期); 关键看'次日1d'有没有预测力(ICIR>0.4且|IC|>0.03)")
