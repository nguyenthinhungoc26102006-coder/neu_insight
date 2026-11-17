"""Three linked analytical solutions with explicit per-stage results."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .config import SETTINGS,Settings
from .engine import read_grade_csv, clean_grades,add_prior_features,read_schedule
from .modeling import out_of_time_predictions
from .detection import detect_sections
from .remediation import adjust_grades
from .tci import compute_tci

class Result:
    def __init__(self,grades,sections,students,summary,schedule,metrics):
        self.grades=grades
        self.sections=sections
        self.students=students
        self.summary=summary
        self.schedule=schedule
        self.metrics=metrics

def _hist(series,bins=np.arange(0,4.01,.25)):
    v=pd.to_numeric(series,errors="coerce").dropna().to_numpy()
    counts,_=np.histogram(v,bins=bins)
    return [{"range":f"{bins[i]:.2f}–{bins[i+1]:.2f}","count":int(x)} for i,x in enumerate(counts)]

def analyze(grade_bytes:bytes,schedule_bytes:bytes|None=None,config:Settings=SETTINGS,on_stage=None)->Result:
    started=time.perf_counter()
    def report(stage):
        if on_stage:on_stage(stage)
    report("Kiểm tra và chuẩn hóa dữ liệu")
    raw=read_grade_csv(grade_bytes)
    base,quality=clean_grades(raw)
    if base.score.notna().sum()<100:raise ValueError("Cần ít nhất 100 bản ghi có điểm HP hợp lệ.")
    if base.section.nunique()<3:raise ValueError("Cần ít nhất 3 lớp học phần để so sánh.")
    report("Xây dựng lịch sử học tập theo từng kỳ")
    base=add_prior_features(base)
    report("Huấn luyện CatBoost/Ridge bằng dữ liệu các năm trước")
    predicted,model_metrics=out_of_time_predictions(base)
    report("Phát hiện bất thường bằng 4 nhóm tín hiệu")
    sections=detect_sections(predicted,config)
    report("Hiệu chỉnh có kiểm soát bằng Empirical Bayes")
    grades,sections,eb_metrics=adjust_grades(predicted,sections,config)
    report("Tính TCI và phân loại sinh viên")
    grades,students,tci_metrics=compute_tci(grades,config)
    schedule=read_schedule(schedule_bytes)
    flagged=sections[sections.anomaly_label.isin(["HIGH_INFLATION","UNUSUALLY_STEEP","SUSPICIOUS_DISTRIBUTION"])]
    summary={
        "dataset":quality,
        "sections_graded":int((sections.n_grade>0).sum()),
        "flagged_sections":int(len(flagged)),
        "label_counts":{str(k):int(v) for k,v in sections.anomaly_label.value_counts().items()},
        "proposed_sections":eb_metrics["sections_proposed"],
        "proposed_grades":eb_metrics["total_adjusted_records"],
        "students_with_tci":tci_metrics["students_with_tci"],
        "provisional_students":tci_metrics["provisional_students"],
        "changed_categories":tci_metrics["changed_categories"],
        "mean_gpa_before":tci_metrics["mean_before"],
        "mean_tci":tci_metrics["mean_after"],
        "gpa_hist_before":_hist(students.gpa_before),
        "gpa_hist_after":_hist(students.tci),
        "models":model_metrics,
        "remediation":eb_metrics,
        "tci":tci_metrics,
        "settings":config.json(),
        "seconds":round(time.perf_counter()-started,2),
        "warnings":[
            "Dữ liệu BTC chỉ gồm các học phần của một khoa. TCI không phải GPA toàn trường.",
            "Mã học phần/lớp trong thời khóa biểu không nối trực tiếp với bảng điểm.",
            "Các ngưỡng và quy đổi thang 4 là giả định phân tích, cần kiểm chứng và xác nhận.",
            "Anomaly Score không phải xác suất gian lận; không thay điểm chính thức."]
    }
    report("Đã hoàn thành")
    return Result(grades,sections,students,summary,schedule,{"model":model_metrics,"eb":eb_metrics,"tci":tci_metrics})
