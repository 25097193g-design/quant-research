"""大宗交易(block_trade) kill-or-keep — rqdatac可用、不受东财IP封, 机构大额撮合(比主力净流入干净)
信号(每月): 大宗笔数bcnt / 大宗金额btot / 折溢价prem(大宗价/收盘-1, 折价=机构出货看空)
口径: 全PIT主板, 2023-08~2026-07, 截面rank IC vs 前向1/3月 | 输出 block_trade_ic.csv
运行: cd projects/week1_capital_flow && conda run -n rqsdk --no-capture-output python -u block_trade_ic.py
"""
import rqdatac as rq; rq.init()
import pandas as pd, numpy as np
DATA='/Users/lichanghaodemac/quant_strategy/data'
OUT='/Users/lichanghaodemac/quant_strategy/projects/week1_capital_flow'

price=pd.read_parquet(f'{DATA}/mainboard_price.parquet')
price['date']=pd.to_datetime(price['date'])
uni=sorted(price['order_book_id'].unique())
print(f"universe {len(uni)}股, 分块拉大宗交易 2023-08~2026-07...",flush=True)
pieces=[]
for i in range(0,len(uni),500):
    ch=uni[i:i+500]
    try:
        d=rq.get_block_trade(ch,'2023-08-01','2026-07-30')
        if d is not None and len(d): pieces.append(d.reset_index())
    except Exception as e: print(f"  chunk{i} ERR {str(e)[:50]}",flush=True)
bt=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
if len(bt)==0: print("无大宗交易数据"); import sys; sys.exit()
bt['trade_date']=pd.to_datetime(bt['trade_date']); bt['ym']=bt['trade_date'].dt.to_period('M')
# 折溢价: 合并收盘价
cl=price[['order_book_id','date','close']]
bt=bt.merge(cl, left_on=['order_book_id','trade_date'], right_on=['order_book_id','date'], how='left', suffixes=('','_cl'))
bt['premium']=(bt['price']/bt['close']-1)
print(f"大宗交易 {len(bt)}条, 涉及{bt['order_book_id'].nunique()}股 | 折溢价中位{bt['premium'].median():.3f}",flush=True)
# 每股每月
g=bt.groupby(['order_book_id','ym']).agg(bcnt=('total_turnover','count'),btot=('total_turnover','sum'),prem=('premium','mean')).reset_index()
# 前向收益
clw=price.pivot(index='date',columns='order_book_id',values='close').sort_index()
mec=clw.resample('ME').last(); mec.index=mec.index.to_period('M')
fwd={k:(mec.shift(-k)/mec-1) for k in [1,3]}
def ic(col,ret):
    sig=g.pivot_table(index='ym',columns='order_book_id',values=col)
    out={}
    for ym in sig.index.intersection(ret.index):
        s=sig.loc[ym].dropna(); r=ret.loc[ym].reindex(s.index).dropna()
        idx=s.index.intersection(r.index)
        if len(idx)<15: continue
        out[ym]=pd.Series(s.reindex(idx)).rank().corr(pd.Series(r.reindex(idx)).rank())
    return pd.Series(out)
rows=[]
for col in ['bcnt','btot','prem']:
    for k in [1,3]:
        s=ic(col,fwd[k]).dropna()
        rows.append({'信号':col,'horizon':f'{k}m','n月':len(s),
            'IC均值':round(s.mean(),4) if len(s) else None,
            'ICIR':round(s.mean()/s.std(),2) if len(s)>2 and s.std()>0 else None,
            '正月比':round((s>0).mean(),2) if len(s) else None,
            '截面中位':int(g.groupby('ym').size().median())})
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/block_trade_ic.csv',index=False)
print("\n=== 大宗交易 截面IC (kill-or-keep) ===")
print(R.to_string(index=False))
print("\n读法: prem负IC=折价出货看空(折价多的跌); bcnt/btot看机构活跃度有无预测力; |IC|≥0.03且ICIR≥0.4才算信号")
