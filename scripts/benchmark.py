"""Stress-test on synthetic data ONLY. Example: python scripts/benchmark.py --copies 18"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.pipeline import analyze

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--copies',type=int,default=18)
    args=ap.parse_args()
    if args.copies<1 or args.copies>100:raise ValueError('copies: 1-100')
    base=pd.read_csv(ROOT/'sample'/'demo_synthetic.csv',encoding='utf-8-sig')
    blocks=[]
    for i in range(args.copies):
        b=base.copy()
        b['Mã SV']=f'B{i:03d}-'+b['Mã SV']
        b['Mã lớp học phần']=f'B{i:03d}-'+b['Mã lớp học phần']
        # deterministic tiny perturbations avoid exact distribution repeats
        rng=np.random.default_rng(5000+i)
        b['Điểm HP (thang 10)']=(pd.to_numeric(b['Điểm HP (thang 10)'],errors='coerce')+rng.normal(0,.05,len(b))).clip(0,10).round(1)
        blocks.append(b)
    raw=pd.concat(blocks,ignore_index=True).to_csv(index=False).encode('utf-8-sig')
    start=time.perf_counter()
    result=analyze(raw,on_stage=lambda s:print(f'{time.perf_counter()-start:.1f}s {s}',flush=True))
    print(json.dumps({'rows':len(base)*args.copies,'seconds':result.summary['seconds'],
                      'sections':result.summary['dataset']['section_count'],
                      'students':result.summary['dataset']['student_count'],
                      'flagged':result.summary['flagged_sections']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
