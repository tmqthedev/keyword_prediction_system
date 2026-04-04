KEYWORD SUGGESTION SYSTEM

1) Yeu cau moi truong
- Khuyen nghi Python 3.11.
- Khong dung Python 3.13 voi bo thu vien hien tai.

2) Cai dat nhanh (PowerShell)
Chay trong thu muc du an:

	C:/Users/Lenovo/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
	.\.venv\Scripts\Activate.ps1
	New-Item -ItemType Directory -Force D:\Temp | Out-Null
	$env:TEMP='D:\Temp'; $env:TMP='D:\Temp'
	pip install --no-cache-dir -r requirements.txt

3) Chay backend API
	uvicorn app:app --reload

API mac dinh: http://127.0.0.1:8000
Health check: http://127.0.0.1:8000/health

4) Chay giao dien UI
- Mo file frontend.html bang Live Server hoac mo truc tiep tren trinh duyet.
- UI goi API toi http://127.0.0.1:8000

5) Train model (tuy chon)
	.\.venv\Scripts\python.exe training.py

Ket qua sau train:
- Model luu tai ./final_model
- Checkpoint luu tai ./results
- Mapping nhan luu tai ./final_model/label_mapping.json

6) Cac file chinh
- app.py: FastAPI backend cho UI
- frontend.html: giao dien nguoi dung
- training.py: pipeline train BERT tu data.csv
- data.csv: du lieu co 2 cot bat buoc: query, keyword

7) Xu ly su co thuong gap
- Loi pip khi cai dat tren Python 3.13: dung Python 3.11
- Loi het dung luong o C:: dat TEMP/TMP sang D:\Temp va dung --no-cache-dir