"""Local single-workstation FastAPI dashboard. Do NOT expose raw student files publicly."""
from __future__ import annotations
import io,json,threading,time,uuid,zipfile
from pathlib import Path
import pandas as pd
from fastapi import FastAPI,File,UploadFile,HTTPException,Query
from fastapi.responses import FileResponse,StreamingResponse
from fastapi.staticfiles import StaticFiles
from .pipeline import analyze,Result

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"output"
OUT.mkdir(exist_ok=True)
app=FastAPI(title="NEU Grade Intelligence",version="1.0.0",description="3-stage academic analytics; local-only demonstration")
app.mount("/assets",StaticFiles(directory=str(ROOT/"web")),name="assets")
app.state.job={"state":"EMPTY","stage":"Chưa có dữ liệu","id":None}
app.state.result=None
_lock=threading.RLock()


def _records(df:pd.DataFrame):
    return json.loads(df.to_json(orient="records",date_format="iso",force_ascii=False,double_precision=6))

def _run(jobid,grade_bytes,schedule_bytes):
    def stage(msg):
        with _lock:
            app.state.job.update(state="RUNNING",stage=msg)
    try:
        result=analyze(grade_bytes,schedule_bytes,on_stage=stage)
        with _lock:
            app.state.job.update(state="RUNNING",stage="Đang lưu các báo cáo CSV")
        # Output stays local and out of the downloadable source archive.
        for name,frame in [("anomalies",result.sections),("adjustments",result.grades),
                           ("tci",result.students)]:
            frame.to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
        (OUT/"summary.json").write_text(json.dumps(result.summary,ensure_ascii=False,indent=2),encoding="utf-8")
        with _lock:
            app.state.result=result
            app.state.job.update(state="DONE",stage="Hoàn thành",seconds=result.summary["seconds"])
    except Exception as exc:
        with _lock:
            app.state.job.update(state="FAILED",stage="Phân tích thất bại",error=str(exc))


def _result()->Result:
    result=app.state.result
    if result is None:raise HTTPException(409,"Chưa có kết quả. Hãy tải dữ liệu và chạy phân tích.")
    return result

@app.get("/")
def index():return FileResponse(ROOT/"web"/"index.html")

@app.get("/api/job")
def job():return app.state.job

@app.post("/api/analyze")
async def start(grades:UploadFile=File(...),schedule:UploadFile|None=File(None)):
    with _lock:
        if app.state.job["state"]=="RUNNING":raise HTTPException(409,"Hệ thống đang xử lý dữ liệu.")
    raw=await grades.read()
    if not raw or len(raw)>80*1024*1024:raise HTTPException(400,"CSV rỗng hoặc quá 80 MB.")
    schedule_raw=await schedule.read() if schedule else None
    jobid=uuid.uuid4().hex[:12]
    with _lock:
        app.state.job={"state":"RUNNING","stage":"Đã nhận dữ liệu, bắt đầu phân tích","id":jobid}
        app.state.result=None
    threading.Thread(target=_run,args=(jobid,raw,schedule_raw),daemon=True).start()
    return {"job_id":jobid,"status":"RUNNING"}

@app.post("/api/demo")
def start_demo():
    with _lock:
        if app.state.job["state"]=="RUNNING":raise HTTPException(409,"Hệ thống đang xử lý.")
    demo=(ROOT/"sample"/"demo_synthetic.csv").read_bytes()
    jobid=uuid.uuid4().hex[:12]
    with _lock:
        app.state.job={"state":"RUNNING","stage":"Chạy dữ liệu GIẢ LẬP, không phải BTC","id":jobid}
        app.state.result=None
    threading.Thread(target=_run,args=(jobid,demo,None),daemon=True).start()
    return {"job_id":jobid,"status":"RUNNING"}

@app.get("/api/summary")
def summary():return _result().summary

@app.get("/api/schedule")
def schedule():return _result().schedule

@app.get("/api/anomalies")
def anomalies(limit:int=Query(200,ge=1,le=1500),label:str="ALL",course:str="",year:str=""):
    d=_result().sections
    if label!="ALL":d=d[d.anomaly_label==label]
    if course:d=d[d.course.astype(str).str.contains(course,case=False,regex=False)]
    if year:d=d[d.year.astype(str)==year]
    return {"count":len(d),"items":_records(d.head(limit))}

@app.get("/api/sections/{section_id}")
def section(section_id:str):
    r=_result()
    subset=r.sections[r.sections.section==section_id]
    if subset.empty:raise HTTPException(404,"Không tìm thấy lớp học phần")
    grades=r.grades[r.grades.section==section_id]
    cols=["student","score","expected","adjusted_score","gpa_before","gpa_after",
          "pi_low","pi_high","individual_review","record_status","prior_count"]
    return {"section":_records(subset)[0],"grades":_records(grades[cols].sort_values("student"))}

@app.get("/api/students")
def students(limit:int=Query(150,ge=1,le=5000),query:str="",classification:str="ALL",confidence:str="ALL"):
    d=_result().students
    if query:d=d[d.student.str.contains(query,case=False,regex=False)]
    if classification!="ALL":d=d[d.classification_after==classification]
    if confidence!="ALL":d=d[d.confidence==confidence]
    return {"count":len(d),"items":_records(d.head(limit))}

@app.get("/api/students/{student_id}")
def student(student_id:str):
    r=_result()
    entry=r.students[r.students.student==student_id]
    if entry.empty:raise HTTPException(404,"Không tìm thấy sinh viên")
    d=r.grades[r.grades.student==student_id]
    cols=["section","course","year","semester","credits","project","score","expected",
          "adjusted_score","gpa_before","gpa_after","tci_contribution","section_status"]
    return {"student":_records(entry)[0],"courses":_records(d[cols].sort_values(["year","semester","course"]))}

@app.get("/api/transitions")
def transitions():
    d=_result().students.groupby(["classification_before","classification_after"]).size().reset_index(name="count")
    return {"items":_records(d)}

@app.get("/api/download/{name}")
def download(name:str):
    if name not in ("anomalies","adjustments","tci","summary"):
        raise HTTPException(404,"Báo cáo không tồn tại")
    _result()
    file=OUT/(f"{name}.json" if name=="summary" else f"{name}.csv")
    if not file.exists():raise HTTPException(409,"Đang ghi báo cáo, thử lại.")
    return FileResponse(file,filename=file.name,media_type="application/octet-stream")

@app.get("/api/health")
def health():return {"ok":True,"mode":"local_only","has_results":app.state.result is not None}
