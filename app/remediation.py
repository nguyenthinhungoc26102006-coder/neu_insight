"""Empirical-Bayes location shift, with abstention and split-style conformal residual bands."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Settings

DIRECTIONAL={"HIGH_INFLATION","UNUSUALLY_STEEP"}

def _quantile_higher(a,frac):
    a=np.sort(np.asarray(a,dtype=float))
    if not len(a):return np.nan
    # Finite-sample conformal quantile: ceil((n+1)*coverage) / n.
    j=int(np.ceil((len(a)+1)*frac))-1
    return float(a[max(0,min(j,len(a)-1))])

def adjust_grades(df:pd.DataFrame,secs:pd.DataFrame,config:Settings)->tuple[pd.DataFrame,pd.DataFrame,dict]:
    sections=secs.copy()
    decisions=[];shifts=[];explanations=[]
    for _,s in sections.iterrows():
        is_directional=s.anomaly_label in DIRECTIONAL
        enough=(s.delta_method=="MODEL_RESIDUAL" and s.n_prediction>=config.min_predicted_rows
             and s.peer_sections>=config.min_peer_sections and s.evidence_quality in ("MEDIUM","HIGH"))
        shift=float(s.eb_shift) if pd.notna(s.eb_shift) else 0.0
        post_sd=float(s.eb_posterior_sd) if pd.notna(s.eb_posterior_sd) else float('inf')
        # Show proposals only if directional, sufficiently evidenced, and posterior shift discernible.
        supported=(is_directional and enough and abs(shift)>0.35 and abs(shift)>1.645*post_sd)
        if supported:
            approved_shift=float(np.clip(shift,-config.max_adjustment,config.max_adjustment))
            state="ADJUSTMENT_PROPOSED"
            note="EB: dịch điểm ở cấp lớp; không thay đổi điểm học vụ chính thức. Cần người có thẩm quyền thẩm định."
        elif is_directional or s.anomaly_label in ("SUSPICIOUS_DISTRIBUTION","DATA_QUALITY_REVIEW"):
            approved_shift=0.0; state="REVIEW_REQUIRED"
            note="Có tín hiệu đáng xem xét nhưng chưa đủ căn cứ để áp dụng phép dịch điểm."
        else:
            approved_shift=0.0;state="NO_ADJUSTMENT"
            note="Không phát hiện tín hiệu đủ điều kiện hoặc không có nhóm tham chiếu."
        decisions.append(state);shifts.append(approved_shift);explanations.append(note)
    sections["adjustment_status"]=decisions
    sections["applied_shift"]=shifts
    sections["adjustment_note"]=explanations
    lookup=sections.set_index("section")
    grades=df.copy()
    grades["section_status"]=grades.section.map(lookup.adjustment_status)
    grades["section_shift"]=grades.section.map(lookup.applied_shift).fillna(0.0)
    grades["raw_score"]=grades.score
    # Zero may mean absent; never silently correct it. Missing score remains missing.
    grades["adjusted_score"]=grades.score
    eligible=grades.score.notna() & (grades.score>0) & (grades.section_status=="ADJUSTMENT_PROPOSED")
    grades.loc[eligible,"adjusted_score"]=(grades.loc[eligible,"score"]-grades.loc[eligible,"section_shift"]).clip(0,10)
    grades["grade_change"]=grades.adjusted_score-grades.score
    # Calibration uses OOT predictions only, on relatively unflagged sections.
    calib_sections=set(sections.loc[sections.anomaly_label=="NORMAL","section"])
    cal=grades.loc[grades.section.isin(calib_sections)&grades.expected.notna()&grades.score.notna()].copy()
    if len(cal)<100:
        cal=grades.loc[grades.expected.notna()&grades.score.notna()].copy()
    global_abs=(cal.score-cal.expected).abs().to_numpy()
    global_radius=_quantile_higher(global_abs,.95)
    if not np.isfinite(global_radius):global_radius=np.nan
    radii={}
    for group,part in cal.groupby("group"):
        if len(part)>=100:
            radii[group]=_quantile_higher((part.score-part.expected).abs().to_numpy(),.95)
    grades["pi_radius"]=grades.group.map(radii).fillna(global_radius)
    grades["pi_low"]=(grades.expected-grades.pi_radius).clip(0,10)
    grades["pi_high"]=(grades.expected+grades.pi_radius).clip(0,10)
    grades["individual_review"]=(grades.score.notna() & grades.expected.notna() &
        ((grades.score<grades.pi_low)|(grades.score>grades.pi_high)))
    grades.loc[grades.score==0,"individual_review"]=False
    grades["record_status"]=np.where(grades.score.isna(),"MISSING_GRADE",
        np.where(grades.score==0,"ZERO_SCORE_UNRESOLVED",
        np.where(grades.individual_review,"REVIEW_REQUIRED",grades.section_status)))
    metrics={"total_adjusted_records":int(eligible.sum()),
        "sections_proposed":int((sections.adjustment_status=="ADJUSTMENT_PROPOSED").sum()),
        "sections_for_review":int((sections.adjustment_status=="REVIEW_REQUIRED").sum()),
        "individual_review":int(grades.individual_review.sum()),
        "conformal_radius_global":round(float(global_radius),4) if np.isfinite(global_radius) else None,
        "conformal_calibration_count":int(len(cal)),
        "note":"Khoảng dự báo dùng OOT residual trên lớp tương đối bình thường; mức phủ 95% mang tính ước lượng, không phải bảo đảm cho từng nhóm."}
    return grades,sections,metrics
