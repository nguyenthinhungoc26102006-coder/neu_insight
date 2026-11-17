from fastapi.testclient import TestClient
from app.main import app
import time

client=TestClient(app)

def test_ui_and_demo_analysis():
    r=client.get('/')
    assert r.status_code==200 and 'NEU' in r.text
    assert client.get('/assets/styles.css').status_code==200
    response=client.post('/api/demo')
    assert response.status_code==200
    for _ in range(120):
        s=client.get('/api/job').json()
        if s['state'] in ('DONE','FAILED'):break
        time.sleep(.2)
    assert s['state']=='DONE',s
    summary=client.get('/api/summary').json()
    assert summary['dataset']['section_count']==192
    an=client.get('/api/anomalies?limit=10').json()
    assert an['items']
    sec=an['items'][0]['section']
    assert client.get(f'/api/sections/{sec}').status_code==200
    st=client.get('/api/students?limit=1').json()['items'][0]['student']
    assert client.get(f'/api/students/{st}').status_code==200
    assert client.get('/api/download/tci').status_code==200
    assert client.get('/api/transitions').status_code==200
