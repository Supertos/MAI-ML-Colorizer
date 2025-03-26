## Запуск
```bash
git clone -b backend https://github.com/Supertos/MAI-ML-Colorizer.git
pip install -r requirements.txt
cd app
python -m celery -A celery_app.tasks worker --loglevel=INFO --pool=solo
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
