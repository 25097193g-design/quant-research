"""融资融券 kill-or-keep: 建3年panel(按日拉沪深 month-end × 全截面) + 截面rank IC vs 前向收益
信号: 融资余额(mqy)水平、1月变化(Δmqy=杠杆资金流入)、融券余量(ryl 空头) | 前向1/3月收益
输出: margin_panel.parquet, margin_ic.csv | 运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python build_margin_panel_ic.py
"""
import akshare as ak, pandas as pd, numpy as np, time, threading
def call_timeout(fn,t=20):
    box=[None]
    def _r():
        try: box[0]=fn()
        except Exception: box[0]=None
    th=threading.Thread(target=_r,daemon=True); th.start(); th.join(t)
    return box[0]
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'

price=pd.read_parquet(f'{DATA}/mainboard_price.parquet')
price['date']=pd.to_datetime(price['date'])
dates=pd.DatetimeIndex(sorted(price['date'].unique()))
me=dates.to_series().resample('ME').last()
me=[d for d in me if pd.notna(d) and d>=pd.Timestamp('2025-08-01')]
print(f"拉融资融券 {len(me)}个月末(沪深), 每个约2调用...")

def retry(fn,n=2):
    for i in range(n):
        try: return fn()
        except Exception: time.sleep(0.8)
    return None
def pick(df,d):
    cc=next((c for c in df.columns if '代码' in c),None)
    mqy=next((c for c in df.columns if '融资余额' in c),None)
    ryl=next((c for c in df.columns if '融券余量' in c),None)
    if not(cc and mqy): return None
    out=pd.DataFrame({'code':df[cc].astype(str).str.zfill(6),'mqy':pd.to_numeric(df[mqy],errors='coerce')})
    if ryl: out['ryl']=pd.to_numeric(df[ryl],errors='coerce')
    out['date']=d; return out

pieces=[]
for k,d in enumerate(me):
    ds=d.strftime('%Y%m%d')
    for fn in ['stock_margin_detail_sse','stock_margin_detail_szse']:
        df=call_timeout(lambda f=fn,s=ds: retry(lambda: getattr(ak,f)(s)),t=15)
        if df is None or len(df)==0: continue
        p=pick(df,d)
        if p is not None: pieces.append(p)
    time.sleep(2.0)
    print(f"  {k+1}/{len(me)} {d.date()}",flush=True)
panel=pd.concat(pieces,ignore_index=True)
panel=panel.dropna(subset=['mqy']); panel=panel[panel['mqy']>0]
panel['order_book_id']=panel['code'].map(lambda c: f'{c}.XSHG' if c.startswith('6') else f'{c}.XSHE')
panel['ym']=panel['date'].dt.to_period('M')
panel.to_parquet(f'{OUT}/margin_panel.parquet',index=False)
print(f"panel: {panel.shape} | 股{panel['order_book_id'].nunique()} 月{panel['ym'].nunique()}")

# 前向收益
close=price.pivot(index='date',columns='order_book_id',values='close').sort_index()
mec=close.resample('ME').last(); mec.index=mec.index.to_period('M')
fwd={k:(mec.shift(-k)/mec-1) for k in [1,3]}
# 信号 wide
mqy=panel.pivot_table(index='ym',columns='order_book_id',values='mqy',aggfunc='last')
ryl=panel.pivot_table(index='ym',columns='order_book_id',values='ryl',aggfunc='last') if 'ryl' in panel.columns else None
dmqy=mqy.diff(1)  # 1月融资余额变化
def ic(sig,ret):
    out={}
    for ym in sig.index.intersection(ret.index):
        s=sig.loc[ym].dropna(); r=ret.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<30: continue
        out[str(ym)]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    return pd.Series(out)
rows=[]
for name,sig in [('融资余额_水平',mqy),('Δ融资余额_1月',dmqy)]+([('融券余量',ryl)] if ryl is not None else []):
    for k in [1,3]:
        s=ic(sig,fwd[k]).dropna()
        if len(s)<6: continue
        rows.append({'信号':name,'horizon':f'{k}m','n月':len(s),'IC均值':round(s.mean(),4),'ICIR':round(s.mean()/s.std(),2) if s.std()>0 else 0,'正月比':round((s>0).mean(),2)})
R=pd.DataFrame(rows)
R.to_csv(f'{OUT}/margin_ic.csv',index=False)
print("\n=== 融资融券 截面IC (近3年, kill-or-keep) ===")
print(R.to_string(index=False))
print("\n读法: IC|≥0.03且ICIR≥0.4才算有信号; 正负号看方向(正=融资多的涨/负=融资多的跌)")
