"""Credit-weighted observed-scope GPA/TCI; exact additive attribution of changes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Settings

def grade_point(x,config:Settings):
    if pd.isna(x):return np.nan
    for min_score,gp in config.grade_scale:
        if float(x)>=min_score:return float(gp)
    return 0.0

def label_tci(tci):
    if pd.isna(tci):return "NOT_AVAILABLE"
    if tci>=3.6:return "ADVANCED"
    if tci>=3.2:return "PROFICIENT"
    if tci>=2.5:return "INTERMEDIATE"
    return "AT_RISK"

def compute_tci(grades:pd.DataFrame,config:Settings)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    g=grades.copy()
    g["gpa_before"]=g.score.map(lambda x:grade_point(x,config))
    g["gpa_after"]=g.adjusted_score.map(lambda x:grade_point(x,config))
    g["valid_tci"]=g.score.notna()&(g.credits>0)
    g["weighted_before"]=np.where(g.valid_tci,g.gpa_before*g.credits,0.0)
    g["weighted_after"]=np.where(g.valid_tci,g.gpa_after*g.credits,0.0)
    g["tci_change_numerator"]=g.weighted_after-g.weighted_before
    valid=g[g.valid_tci].copy()
    if valid.empty:
        return g,pd.DataFrame(),{"students_with_tci":0}
    s=valid.groupby("student",sort=False).agg(
        n_courses=("section","size"),sum_credits=("credits","sum"),
        before_sum=("weighted_before","sum"), after_sum=("weighted_after","sum"),
        flagged_courses=("section_status",lambda x:int((x=="ADJUSTMENT_PROPOSED").sum())),
        project_courses=("project",lambda x:int((x=="Có").sum())),
        cohort=("cohort","first"),faculty=("faculty","first"),
        year_last=("year_start","max")
    ).reset_index()
    s["gpa_before"]=(s.before_sum/s.sum_credits).round(4)
    s["tci"]=(s.after_sum/s.sum_credits).round(4)
    s["delta_tci"]=(s.tci-s.gpa_before).round(4)
    s["classification_before"]=s.gpa_before.map(label_tci)
    s["classification_after"]=s.tci.map(label_tci)
    s["confidence"] = np.where(s.n_courses<config.min_tci_courses,"PROVISIONAL","OBSERVED_SCOPE")
    bounds=np.array([2.5,3.2,3.6])
    s["near_boundary"]=s.tci.map(lambda x:bool(np.min(abs(bounds-x))<.10))
    no_project=valid[valid.project!="Có"].groupby("student").agg(
        non_project_credits=("credits","sum"),non_project_points=("weighted_after","sum"))
    s=s.merge(no_project,on="student",how="left")
    s["tci_coursework"]=(s.non_project_points/s.non_project_credits).round(4)
    s["tci_coursework"]=s.tci_coursework.replace([np.inf,-np.inf],np.nan)
    # Exact contribution decomposition on the SAME observed courses.
    denom=s.set_index("student").sum_credits
    g["tci_contribution"]=(g.tci_change_numerator/g.student.map(denom)).fillna(0)
    s["rank_in_sample"]=s.tci.rank(method="min",ascending=False).astype(int)
    s=s.sort_values("tci",ascending=False).reset_index(drop=True)
    metrics={"students_with_tci":int(len(s)),"provisional_students":int((s.confidence=="PROVISIONAL").sum()),
        "mean_before":round(float(s.gpa_before.mean()),4),"mean_after":round(float(s.tci.mean()),4),
        "changed_categories":int((s.classification_before!=s.classification_after).sum()),
        "note":"TCI = GPA hiệu chỉnh chỉ trên các học phần BTC cung cấp; quy đổi thang 4 trong cấu hình là giả định, không phải quy chế trường."}
    return g,s,metrics
