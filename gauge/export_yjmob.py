#!/usr/bin/env python3
"""Export YJMob100K (dataset 1) to the parquet file that the question generator reads.

Takes the first 20,000 people in the file (reading at most 34,000,000 rows) and drops anyone with
fewer than 100 records before day 60 or fewer than 50 records from day 60 on; none of the 20,000
is dropped. Each output row is one observation with the columns uid, slot (day * 48 + timeslot),
rank (the place's visit-frequency rank within that person's days before day 60, capped at 511, with
512 for places not seen then), x and y (grid coordinates), n_train_cells and is_test (day >= 60).
The generator draws every question from days before day 60.

YJMob100K is anonymised, discretised to 500 m cells and 30-minute bins, with dates masked, and the
data provider does not disclose the city (Yabe et al. 2024, Scientific Data 11:397). No place is
mapped, geocoded or named. Coordinates enter the benchmark only as relative distances, under a
per-question rigid transform.

Input: data/yjmob100k/yjmob100k-dataset1.csv.gz (download from Zenodo, DOI 10.5281/zenodo.10836269).
Output: the parquet file and a manifest next to it (the same name ending in _manifest.json).

Usage:
  python gauge/export_yjmob.py --out data/yjmob/yjmob_export.parquet
"""
import argparse, json, os
import numpy as np, pandas as pd

SRC=os.path.expanduser("data/yjmob100k/yjmob100k-dataset1.csv.gz")
SPLIT_DAY=60; K_RANKS=512; OOV=K_RANKS

ap=argparse.ArgumentParser()
ap.add_argument("--users",type=int,default=20000)
ap.add_argument("--max-rows",type=int,default=34_000_000)
ap.add_argument("--out",default="data/yjmob/yjmob_export.parquet")
a=ap.parse_args()

print(f"reading up to {a.max_rows:,} rows ...",flush=True)
d=pd.read_csv(SRC,nrows=a.max_rows,dtype=np.int32)
keep=d.uid.unique()[:a.users]
d=d[d.uid.isin(set(keep))].sort_values(["uid","d","t"])
d["slot"]=d.d*48+d.t
d["cell"]=d.x*1000+d.y
print(f"{d.uid.nunique():,} users | {len(d):,} rows",flush=True)

out=[]; drops=dict(short_train=0,short_test=0,kept=0)
for uid,g in d.groupby("uid",sort=False):
    tr=g[g.d<SPLIT_DAY]
    if len(tr)<100: drops["short_train"]+=1; continue
    if (g.d>=SPLIT_DAY).sum()<50: drops["short_test"]+=1; continue
    vc=tr.cell.value_counts()
    rk={c:min(i,K_RANKS-1) for i,c in enumerate(vc.index)}
    out.append(pd.DataFrame(dict(
        uid=np.int32(uid), slot=g.slot.to_numpy(np.int32),
        rank=np.array([rk.get(c,OOV) for c in g.cell],np.int16),
        x=g.x.to_numpy(np.int16), y=g.y.to_numpy(np.int16),
        n_train_cells=np.int16(min(len(vc),32767)),
        is_test=(g.d.to_numpy()>=SPLIT_DAY))))
    drops["kept"]+=1
D=pd.concat(out,ignore_index=True)
os.makedirs(os.path.dirname(a.out),exist_ok=True)
D.to_parquet(a.out,index=False,compression="zstd")
tr=D[~D.is_test]; obs=tr.groupby("uid").size()
meta=dict(users=int(drops["kept"]),rows=int(len(D)),K_RANKS=K_RANKS,SPLIT_DAY=SPLIT_DAY,drops=drops,
          has_xy=True, obs_med=int(obs.median()),
          span_eligible={int(s):int((obs>=s+4).sum()) for s in (64,128,256,512,1024)},
          source_md5="3781f6f03a118b5f639bdb4f94dcfdb8",
          note="ranks + x,y. Coordinates released only under per-item rigid transform. Never map/geocode.")
json.dump(meta,open(a.out.replace(".parquet","_manifest.json"),"w"),indent=1)
print(f"DROPS {drops}")
print(f"wrote {a.out}  {os.path.getsize(a.out)/1e6:.0f} MB | {drops['kept']:,} users")
print("span-eligible:", meta["span_eligible"])
