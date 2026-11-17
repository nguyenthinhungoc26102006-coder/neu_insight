"""Fully synthetic reproducible local demo. NEVER copy private organiser data into the source zip."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
rng=np.random.default_rng(2026)
students=[f"DEMO-S{i:04d}" for i in range(1,241)]
ability={st:float(rng.normal(7.2,1.0)) for st in students}
rows=[];ground_truth={}
for year in range(2019,2025):
  for sem in (1,2):
    for courseid in range(1,9):
      course=f"DEMO-M{courseid:03d}"
      grp="N1" if courseid<=4 else "N2"
      ep="Có" if courseid==8 else "Không"
      diff=(-.6 if courseid in (2,7) else .3 if courseid in (1,6) else 0)
      for part in (1,2):
        sec=f"D{year%100:02d}{sem}{courseid:02d}{part}"
        chosen=rng.choice(students,size=30,replace=False)
        effect=0.0;compression=False
        if year>=2022 and sem==1 and courseid==3 and part==1:effect=2.0
        if year==2023 and sem==2 and courseid==6 and part==2:effect=-2.0
        if year==2024 and sem==2 and courseid==5 and part==1:compression=True
        if effect or compression:ground_truth[sec]={"effect":effect,"compression":compression}
        for st in chosen:
          basic=float(np.clip(ability[st]+diff+rng.normal(0,1.05),0,10))
          test=float(np.clip(basic+rng.normal(-.25,.55),0,10))
          cont=float(np.clip(basic+rng.normal(.8,.5),0,10))
          attend=float(np.clip(9+rng.normal(0,.7),0,10))
          score=.1*attend+.3*cont+.6*test
          score=float(np.clip(score+effect,0,10))
          if compression:score=float(np.clip(8.95+rng.normal(0,.15),0,10))
          score=round(score,1)
          rows.append({"Năm học":f"{year}-{year+1}","Học kỳ":sem,"Mã môn":course,"Nhóm môn":grp,
           "Số TC":3,"Là đồ án/thực tập/khóa luận":"Không","Lớp mang mã EP":ep,
           "Mã lớp học phần":sec,"Sĩ số lớp (trong dữ liệu)":30,"Mã SV":st,"Khóa SV":f"K{year-2019+60}",
           "Khoa của SV":"DEMO-KH01","CC":round(attend,1),"KTTX1":round(cont,1),"KTTX2":np.nan,
           "KTTX (TB)":round(cont,1),"Thi":round(test,1),"Điểm HP (thang 10)":score,
           "Cơ cấu trọng số suy ra (CC/KTTX/Thi, %)":"10/30/60"})
root=Path(__file__).parent
pd.DataFrame(rows).to_csv(root/"demo_synthetic.csv",index=False,encoding="utf-8-sig")
(root/"demo_truth.json").write_text(json.dumps(ground_truth,ensure_ascii=False,indent=2),encoding="utf-8")
print('Synthetic rows:',len(rows),'sections',len(set(r['Mã lớp học phần'] for r in rows)),'injected',ground_truth)
