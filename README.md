git clone -b backend https://github.com/Supertos/MAI-ML-Colorizer.git
pip install -r requirements.txt
cd app
python -m uvicorn app.main:app --reload 
