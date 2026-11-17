"""Input cleaning and time-safe features from the files supplied by the organisers."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd

REQUIRED = ["Năm học", "Học kỳ", "Mã môn", "Nhóm môn", "Số TC",
            "Là đồ án/thực tập/khóa luận", "Lớp mang mã EP", "Mã lớp học phần",
            "Mã SV", "Khóa SV", "Khoa của SV", "CC", "KTTX (TB)",
            "Thi", "Điểm HP (thang 10)"]
SCORE_COLS = ["CC", "KTTX (TB)", "Thi", "Điểm HP (thang 10)"]
CATEGORIES = ["course", "group", "faculty", "cohort", "ep", "project"]


def read_grade_csv(data: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(io.BytesIO(data), encoding="cp1258", low_memory=False)
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError("Thiếu các cột BTC bắt buộc: " + ", ".join(missing))
    if len(frame) > 1_000_000:
        raise ValueError("Giới hạn mỗi lần phân tích là 1.000.000 bản ghi.")
    return frame


def clean_grades(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df = df.rename(columns={
        "Năm học":"year", "Học kỳ":"semester", "Mã môn":"course",
        "Nhóm môn":"group", "Số TC":"credits",
        "Là đồ án/thực tập/khóa luận":"project", "Lớp mang mã EP":"ep",
        "Mã lớp học phần":"section", "Mã SV":"student", "Khóa SV":"cohort",
        "Khoa của SV":"faculty", "CC":"attendance", "KTTX (TB)":"continuous",
        "Thi":"exam", "Điểm HP (thang 10)":"score"
    })
    original_count = len(df)
    for col in ["year","course","group","section","student","cohort","faculty","ep","project"]:
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "UNKNOWN"
    for col in ["attendance", "continuous", "exam", "score", "credits", "semester"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid_scores = {}
    for col in SCORE_COLS:
        mapped = {"CC":"attendance", "KTTX (TB)":"continuous", "Thi":"exam", "Điểm HP (thang 10)":"score"}[col]
        invalid = df[mapped].notna() & ~df[mapped].between(0,10)
        invalid_scores[mapped] = int(invalid.sum())
        df.loc[invalid,mapped] = np.nan
    # Historical term key, no same-semester information in prior features.
    yr = df["year"].str.extract(r"^(\d{4})")[0]
    df["year_start"] = pd.to_numeric(yr,errors="coerce")
    df["semester"] = df["semester"].fillna(0).astype(int)
    df["term"] = df["year_start"].fillna(-1).astype(int)*10 + df["semester"]
    df["credits"] = df["credits"].fillna(0).clip(lower=0,upper=30)
    # Duplicates refer to same student in the same section; never count twice.
    duplic = int(df.duplicated(["student","section"]).sum())
    df = df.drop_duplicates(["student","section"],keep="first").reset_index(drop=True)
    missing_key = (df["student"]=="UNKNOWN")|(df["section"]=="UNKNOWN")
    bad_keys = int(missing_key.sum())
    df = df[~missing_key].copy().reset_index(drop=True)
    # Only claim valid score rows for analysis. Keep score missing for quality reporting.
    for c in CATEGORIES:
        df[c] = df[c].astype(str).fillna("UNKNOWN")
    summary = {
        "raw_rows":original_count, "rows":len(df), "duplicate_rows_removed":duplic,
        "missing_id_rows_removed":bad_keys, "valid_grade_rows":int(df.score.notna().sum()),
        "missing_grade_rows":int(df.score.isna().sum()), "invalid_grade_cells":invalid_scores,
        "missing_components":{c:int(df[c].isna().sum()) for c in ["attendance","continuous","exam"]},
        "course_count":int(df.course.nunique()), "section_count":int(df.section.nunique()),
        "student_count":int(df.student.nunique()), "years":sorted(df.year.unique().tolist())
    }
    return df,summary


def add_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strict semester-before-only history, does not use the current class or future grades."""
    out = df.copy()
    for col in ["prior_count", "prior_mean", "prior_std", "prior_group_mean", "prior_group_count", "prior_credits"]:
        out[col] = np.nan if col in ("prior_mean","prior_std","prior_group_mean") else 0.0
    # Iterate over times; commit current grades to memory only AFTER features are generated.
    history: dict[str,list] = {}
    group_history: dict[tuple,list] = {}
    for term in sorted(out.term.unique()):
        ids = out.index[out.term==term]
        # Fast lookup of prior state; one iteration per unique term, no cross-student leaks.
        sub = out.loc[ids,["student","group"]]
        out.loc[ids,"prior_count"] = [len(history.get(s,[])) for s in sub.student]
        out.loc[ids,"prior_mean"] = [np.mean(history[s]) if history.get(s) else np.nan for s in sub.student]
        out.loc[ids,"prior_std"] = [np.std(history[s],ddof=1) if len(history.get(s,[]))>1 else np.nan for s in sub.student]
        out.loc[ids,"prior_group_count"] = [len(group_history.get((s,g),[])) for s,g in zip(sub.student,sub.group)]
        out.loc[ids,"prior_group_mean"] = [np.mean(group_history[(s,g)]) if group_history.get((s,g)) else np.nan for s,g in zip(sub.student,sub.group)]
        # Aggregate credits through a separate accumulator, prior terms only.
        for row in out.loc[ids,["student","group","score"]].itertuples(index=False):
            if pd.notna(row.score):
                history.setdefault(row.student,[]).append(float(row.score))
                group_history.setdefault((row.student,row.group),[]).append(float(row.score))
    # Independent cumulative credit calculation, excludes current term and future terms.
    temp = out[out.score.notna()][["student","term","credits"]].groupby(["student","term"],as_index=False).credits.sum()
    temp = temp.sort_values(["student","term"])
    temp["prev_credits"] = temp.groupby("student").credits.cumsum() - temp.credits
    # For no-score terms, merge_asof would be needed; use explicit lookup by prior-term index.
    lookup = {}
    for st, part in temp.groupby("student",sort=False):
        lookup[st] = (part.term.to_numpy(),part.credits.cumsum().to_numpy())
    prior_credits=[]
    for st,tm in zip(out.student,out.term):
        z=lookup.get(st)
        if z is None: prior_credits.append(0.0);continue
        loc=np.searchsorted(z[0],tm,side="left")-1
        prior_credits.append(float(z[1][loc]) if loc>=0 else 0.0)
    out["prior_credits"] = prior_credits
    return out


def read_schedule(data:bytes|None) -> dict:
    if not data:
        return {"available": False, "note":"Không tải thời khóa biểu. Không ghép mã TKB với bảng điểm."}
    df = pd.read_csv(io.BytesIO(data),encoding="utf-8-sig",low_memory=False)
    return {
        "available":True, "rows":int(len(df)),
        "distinct_sections":int(df["Mã LHP"].nunique()) if "Mã LHP" in df else None,
        "schedule_types":{str(k):int(v) for k,v in df["Dạng lịch"].value_counts(dropna=False).items()} if "Dạng lịch" in df else {},
        "note":"Mã HP/Mã LHP trong TKB không trùng Mã môn/Mã lớp trong bảng điểm; chỉ thống kê độc lập."
    }
