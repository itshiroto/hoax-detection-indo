FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 7860

CMD ["bash", "-c", "python gradio_app.py & uvicorn hoax_detect.api:app --host 0.0.0.0 --port 8000"]
