"""Hierarchical leave-one-section-out reference, model residuals and interpretable anomaly signals."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm, betabinom
from statsmodels.stats.multitest import multipletests
from .config import Settings


def robust_sd(vals, floor=.35):
    a=pd.to_numeric(pd.Series(vals),errors="coerce").dropna().to_numpy(dtype=float)
    if len(a)<2: return float(floor)
    return float(max(floor, 1.4826*np.median(np.abs(a-np.median(a)))))

def _finite(v):
    return np.isfinite(v) if isinstance(v,(float,int,np.number)) else False

def _sig(z):
    return float(2*norm.sf(abs(z))) if _finite(z) else np.nan

def summarize_sections(df:pd.DataFrame)->pd.DataFrame:
    """Vectorized section statistics, scalable to >100k student-grade rows."""
    g=df.groupby("section",sort=False,observed=True)
    stat=g.agg(course=("course","first"),group=("group","first"),
        year=("year","first"),year_start=("year_start","first"),
        semester=("semester","first"),ep=("ep","first"),
        project=("project","first"),credits=("credits","first"),
        faculty_mode=("faculty","first"),n_total=("student","size"),
        n_grade=("score","count"),exam_mean=("exam","mean"),
        continuous_mean=("continuous","mean"),
        exam_missing=("exam",lambda x:float(x.isna().mean())),
        continuous_missing=("continuous",lambda x:float(x.isna().mean())))
    valid=df[df.score.notna()].copy()
    vg=valid.groupby("section",sort=False,observed=True)
    part=vg.agg(mean=("score","mean"),sd=("score","std"),
        a_rate=("score",lambda x:float((x>=8.5).mean())),
        f_rate=("score",lambda x:float((x<4.5).mean())),
        zero_rate=("score",lambda x:float((x==0).mean())),
        prior_valid=("prior_mean","count"),
        n_prediction=("expected","count"),
        residual_mean=("residual","mean"),residual_sd=("residual","std"),
        predictor=("model_name","first"))
    # Grouped quantile on original grade values, not imputation.
    quant=vg.score.quantile([.25,.75]).unstack()
    part["iqr"] = quant[.75]-quant[.25]
    stat=stat.join(part,how="left")
    stat["prior_fraction"]=(stat.prior_valid/stat.n_grade).fillna(0)
    stat["prediction_fraction"]=(stat.n_prediction/stat.n_grade).fillna(0)
    # Pairwise KTTX-Thi is computed only on rows where BOTH components exist.
    pair=valid[valid.continuous.notna()&valid.exam.notna()].copy()
    pair["gap_value"]=pair.continuous-pair.exam
    gp=pair.groupby("section",observed=True).gap_value.agg(gap="mean",gap_n="count")
    stat=stat.join(gp,how="left")
    stat.loc[stat.gap_n.fillna(0)<5,"gap"]=np.nan
    stat["gap_n"]=stat.gap_n.fillna(0).astype(int)
    stat["rho"]=np.nan
    prev=valid[valid.prior_mean.notna()].copy()
    for sec,rows in prev.groupby("section",sort=False,observed=True):
        if len(rows)>=12 and rows.prior_mean.std()>.05 and rows.score.std()>.05:
            stat.loc[sec,"rho"]=float(spearmanr(rows.prior_mean,rows.score).statistic)
    stat["year_start"]=stat.year_start.fillna(-1).astype(int)
    stat["semester"]=stat.semester.fillna(0).astype(int)
    stat["n_prediction"]=stat.n_prediction.fillna(0).astype(int)
    stat["prior_valid"]=stat.prior_valid.fillna(0).astype(int)
    return stat.reset_index()

class ReferenceIndex:
    """Pre-group sections once instead of re-scanning all sections for every class."""
    def __init__(self, sections:pd.DataFrame):
        eligible=sections[sections.n_grade>=8]
        self.k1={k:v for k,v in eligible.groupby(["course","year_start","semester","ep","project"],sort=False)}
        self.k2={k:v for k,v in eligible.groupby(["course","ep","project"],sort=False)}
        self.k3={k:v for k,v in eligible.groupby(["group","year_start","ep","project"],sort=False)}
        self.k4={k:v for k,v in eligible.groupby(["group","ep","project"],sort=False)}

    def peers(self, row:pd.Series)->tuple[pd.DataFrame,str]:
        k=(row.course,row.ep,row.project)
        options=[
            ("CÙNG MÔN · CÙNG KỲ",self.k1.get((row.course,row.year_start,row.semester,row.ep,row.project))),
            ("CÙNG MÔN · GẦN NĂM",self.k2.get(k)),
            ("CÙNG NHÓM · CÙNG NĂM",self.k3.get((row["group"],row.year_start,row.ep,row.project))),
            ("CÙNG NHÓM MÔN",self.k4.get((row["group"],row.ep,row.project)))
        ]
        for i,(label,part) in enumerate(options):
            if part is None:continue
            if i==1:part=part[(part.year_start-row.year_start).abs()<=2]
            out=part[part.section!=row.section]
            if len(out)>=2:return out,label
        # Lightweight empty frame only when no match; diagnostics abstain.
        return pd.DataFrame(),"KHÔNG ĐỦ THAM CHIẾU"

def detect_sections(df:pd.DataFrame, config:Settings)->pd.DataFrame:
    sections=summarize_sections(df)
    # Reference correlations are context indicators only, not independent proofs.
    correlations={}
    for key,part in df[df.score.notna() & df.prior_mean.notna()].groupby("group"):
        if len(part)>=30 and part.score.std()>.02 and part.prior_mean.std()>.02:
            correlations[key]=float(spearmanr(part.prior_mean,part.score).statistic)
    output=[]
    index=ReferenceIndex(sections)
    for _,row in sections.iterrows():
        peers,level=index.peers(row)
        r=row.to_dict()
        r["peer_level"]=level
        r["peer_sections"]=len(peers)
        r["reference_mean"]=float(peers["mean"].median()) if len(peers) else np.nan
        r["reference_a_rate"]=float(peers.a_rate.median()) if len(peers) else np.nan
        r["reference_sd"]=float(peers.sd.median()) if len(peers) else np.nan
        r["reference_gap"]=float(peers.gap.median()) if peers.gap.notna().any() else np.nan
        r["peer_mean_sd"]=robust_sd(peers["mean"],.50) if len(peers) else np.nan
        peer_pred=peers.dropna(subset=["residual_mean"])
        r["peer_prediction_count"]=len(peer_pred)
        r["peer_residual_center"]=float(peer_pred.residual_mean.median()) if len(peer_pred) else np.nan
        tau=robust_sd(peer_pred.residual_mean,.40) if len(peer_pred)>=2 else np.nan
        r["tau"]=tau
        # Residual relative to equally treated peer section residuals (not unadjusted full-school mean).
        usable_model=(r["n_prediction"]>=config.min_predicted_rows and
                    r["prediction_fraction"]>=config.min_prior_fraction and len(peer_pred)>=2)
        if usable_model:
            delta=float(r["residual_mean"]-r["peer_residual_center"])
            residual_sd=r["residual_sd"] if _finite(r["residual_sd"]) else 1.5
            v=float(max(.25, residual_sd)**2/max(1,r["n_prediction"]))
            method="MODEL_RESIDUAL"
        elif len(peers)>=2 and _finite(r["mean"]):
            delta=float(r["mean"]-r["reference_mean"])
            v=float(max(.5,r["sd"] if _finite(r["sd"]) else 1.5)**2 / max(1,r["n_grade"]))
            tau=r["peer_mean_sd"]
            method="RAW_PEER_DIFFERENCE"
        else:
            delta=np.nan;v=np.nan;method="NO_REFERENCE"
        r["delta"]=delta; r["v"]=v;r["delta_method"]=method
        r["z_adj"]=float(delta/math.sqrt(tau*tau+v)) if _finite(delta) and _finite(tau) else np.nan
        r["p_approx"]=_sig(r["z_adj"])
        # Posterior shrinkage only when deltas are based on OOT expected grades.
        if usable_model:
            lam=float(tau**2/(tau**2+v))
            r["eb_lambda"]=lam
            r["eb_shift"]=float(lam*delta)
            r["eb_posterior_sd"]=float(math.sqrt(tau*tau*v/(tau*tau+v)))
        else:
            r["eb_lambda"]=np.nan;r["eb_shift"]=np.nan;r["eb_posterior_sd"]=np.nan
        # Beta-binomial POSTERIOR PREDICTIVE upper-tail probability, not a raw 60% rule.
        if len(peers) and r["n_grade"]:
            ref_a=float(np.clip(np.average(peers.a_rate,weights=peers.n_grade),.025,.975))
            ref_f=float(np.clip(np.average(peers.f_rate,weights=peers.n_grade),.025,.975))
            n=int(r["n_grade"])
            a=int(round(n*r["a_rate"]))
            f=int(round(n*r["f_rate"]))
            t=config.prior_strength_beta
            p_a=float(betabinom.sf(a-1,n,ref_a*t,(1-ref_a)*t))
            p_f=float(betabinom.sf(f-1,n,ref_f*t,(1-ref_f)*t))
        else:
            p_a=np.nan;p_f=np.nan
        r["p_a_tail"]=p_a;r["p_f_tail"]=p_f
        # 4 INTERPRETABLE SIGNALS (0--1), unavailable signals excluded, weights renormalized.
        location=np.clip(.6*min(1,abs(delta)/1.5)+.4*min(1,max(0,abs(r["z_adj"])-.5)/2.5),0,1) if _finite(delta) and _finite(r["z_adj"]) else np.nan
        tail=np.clip(-math.log10(max(1e-9,min(p_a,p_f)))/3,0,1) if _finite(p_a) and _finite(p_f) else np.nan
        dispersion=np.clip((1-r["sd"]/r["reference_sd"]),0,1) if _finite(r["sd"]) and _finite(r["reference_sd"]) and r["reference_sd"]>.05 else np.nan
        rho_ref=correlations.get(r["group"],np.nan)
        correlation=(np.clip((rho_ref-r["rho"])/max(.3,abs(rho_ref)),0,1) if _finite(rho_ref) and rho_ref>.15 and _finite(r["rho"]) else np.nan)
        distribution=float(np.nanmax([dispersion,correlation])) if _finite(dispersion) or _finite(correlation) else np.nan
        gap=np.clip(abs(r["gap"]-r["reference_gap"])/2.5,0,1) if _finite(r["gap"]) and _finite(r["reference_gap"]) else np.nan
        signals={"location":location,"tail":tail,"distribution":distribution,"assessment":gap}
        weights={"location":.40,"tail":.25,"distribution":.20,"assessment":.15}
        ok=[k for k,v in signals.items() if _finite(v)]
        score=float(sum(weights[k]*signals[k] for k in ok)/sum(weights[k] for k in ok)) if ok else np.nan
        r.update({f"signal_{k}":v for k,v in signals.items()})
        r["anomaly_score"]=score
        r["evidence_quality"]=("HIGH" if usable_model and len(peers)>=3 and r["n_grade"]>=20 else
            "MEDIUM" if len(peers)>=2 and r["n_grade"]>=config.min_section_size else "LOW")
        r["data_quality_review"]=(r["n_grade"]<r["n_total"]*.5 or
            (r["exam_missing"]>.90 and r["continuous_missing"]>.90) or
            (r["continuous_missing"]>.90 and _finite(r["exam_mean"]) and _finite(r["mean"])
             and abs(r["mean"]-r["exam_mean"])>1.5))
        output.append(r)
    out=pd.DataFrame(output)
    out["q_approx"]=np.nan
    valid=out.p_approx.notna()
    if valid.any():out.loc[valid,"q_approx"]=multipletests(out.loc[valid,"p_approx"],method="fdr_bh")[1]
    labels=[];reasons=[]
    for _,r in out.iterrows():
        if r.n_grade<config.min_section_size or r.peer_sections<2:
            label="INSUFFICIENT_EVIDENCE"
        elif r.data_quality_review:
            label="DATA_QUALITY_REVIEW"
        elif _finite(r.anomaly_score) and r.anomaly_score>=config.anomaly_flag_threshold:
            direction=(_finite(r.delta) and abs(r.delta)>=config.direction_min_shift and
                       _finite(r.z_adj) and abs(r.z_adj)>=config.direction_min_z and
                       (_finite(r.q_approx) and r.q_approx<=config.review_q_threshold))
            severe_compression=(_finite(r.sd) and r.sd<.40 and _finite(r.reference_sd)
                and r.reference_sd>r.sd*2.3 and _finite(r.iqr) and r.iqr<.50)
            if severe_compression:
                label="SUSPICIOUS_DISTRIBUTION"
            elif direction:
                label="HIGH_INFLATION" if r.delta>0 else "UNUSUALLY_STEEP"
            else:
                label="SUSPICIOUS_DISTRIBUTION"
        else:
            label="NORMAL"
        labels.append(label)
        notes=[]
        if _finite(r.delta):notes.append(f"Điểm lệch kỳ vọng/tham chiếu {r.delta:+.2f}/10")
        if _finite(r.a_rate):notes.append(f"A (≥8,5): {r.a_rate*100:.1f}%")
        if _finite(r.signal_assessment) and r.signal_assessment>.45:notes.append("KTTX–Thi lệch đáng kể so với nhóm")
        if _finite(r.signal_distribution) and r.signal_distribution>.45:notes.append("Phân phối điểm/tương quan khác nhóm")
        if r.evidence_quality=="LOW":notes.append("Bằng chứng còn hạn chế")
        reasons.append("; ".join(notes))
    out["anomaly_label"]=labels
    out["explanation"]=reasons
    out["review_priority"]=np.where(out.anomaly_label.isin(["HIGH_INFLATION","UNUSUALLY_STEEP","SUSPICIOUS_DISTRIBUTION"]),"REVIEW","MONITOR")
    out["priority_order"]=out.anomaly_label.map({"HIGH_INFLATION":0,"UNUSUALLY_STEEP":0,
        "SUSPICIOUS_DISTRIBUTION":0,"DATA_QUALITY_REVIEW":1,"NORMAL":2,
        "INSUFFICIENT_EVIDENCE":3}).fillna(4)
    out=out.sort_values(["priority_order","anomaly_score","n_grade"],ascending=[True,False,False],na_position="last").reset_index(drop=True)
    return out
