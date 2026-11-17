from pathlib import Path
import numpy as np
import pandas as pd
from app.pipeline import analyze
from app.engine import read_grade_csv,clean_grades,add_prior_features
from app.tci import grade_point,label_tci
from app.config import SETTINGS

BASE=Path(__file__).resolve().parents[1]

def test_grade_point_and_ranges():
    assert grade_point(4.4,SETTINGS)==0
    assert grade_point(4.5,SETTINGS)==1
    assert grade_point(9,SETTINGS)==4
    assert label_tci(3.6)=="ADVANCED"
    assert label_tci(3.2)=="PROFICIENT"
    assert label_tci(2.5)=="INTERMEDIATE"
    assert label_tci(2.49)=="AT_RISK"

def test_strict_previous_semester_no_same_term_leakage():
    d,_=clean_grades(read_grade_csv((BASE/"sample"/"demo_synthetic.csv").read_bytes()))
    x=add_prior_features(d)
    for (st,term),sub in x.groupby(["student","term"]):
        old=x[(x.student==st)&(x.term<term)&x.score.notna()]
        assert all(sub.prior_count==len(old))
        if len(old):assert np.allclose(sub.prior_mean,old.score.mean())

def test_complete_synthetic_pipeline():
    r=analyze((BASE/"sample"/"demo_synthetic.csv").read_bytes())
    sec=r.sections.set_index("section")
    assert sec.loc["D231031","anomaly_label"]=="HIGH_INFLATION"
    assert sec.loc["D232062","anomaly_label"]=="UNUSUALLY_STEEP"
    assert sec.loc["D242051","anomaly_label"]=="SUSPICIOUS_DISTRIBUTION"
    assert sec.loc["D242051","adjustment_status"]=="REVIEW_REQUIRED"
    assert r.grades.adjusted_score.between(0,10).all()
    assert (r.grades.loc[r.grades.section_status!="ADJUSTMENT_PROPOSED","score"]==r.grades.loc[r.grades.section_status!="ADJUSTMENT_PROPOSED","adjusted_score"]).all()
    x=r.grades.groupby("student").tci_contribution.sum()
    s=r.students.set_index("student")
    assert np.allclose(x.loc[s.index].values,s.delta_tci.values,atol=.0002)
    assert r.summary["dataset"]["valid_grade_rows"]==5760
    assert r.summary["proposed_sections"]>=1
