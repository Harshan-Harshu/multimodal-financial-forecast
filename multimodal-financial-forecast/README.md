---
title: 📊 Multimodal Financial Forecast
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# 📊 Multimodal Financial Forecast

This app predicts future financial values from time series using an LSTM model. Input sequences or upload CSVs to get started.

## 🔧 Tech Stack
- **Backend**: FastAPI (`app/main.py`)
- **ML Model**: PyTorch LSTM
- **Deployment**: Hugging Face Spaces (Docker)

## 🚀 How to Use
1. Input a JSON-like sequence.
2. Or upload a `.csv` file with numerical sequences.
3. Click “Predict” to see the model forecast.


## 🧠 How It Works

1. **User Input**: 
   - You can enter sequences directly or upload a `.csv` with your own data.

2. **Model Inference**:
   - The LSTM model is loaded from `pipeline/trained_models/`.
   - Input sequences are normalized and passed into the model.
   - The output is a forecast of future values.

3. **Output**:
   - The results are displayed directly on the webpage.
   - Graph removed (clean, console-style output).

---

## 🛠 Tech Stack

| Layer       | Tech        |
|-------------|-------------|
| Backend     | FastAPI     |
| ML Model    | PyTorch LSTM|
| Frontend    | HTML + CSS  |
| Deployment  | Docker + Hugging Face Spaces |

---

## 📂 Project Structure

multimodal-financial-forecast/
├── app/
│ ├── main.py # FastAPI backend
│ ├── templates/
│ │ └── index.html # User interface
│ └── static/
│ └── style.css # Stylesheet
│
├── pipeline/
│ ├── inference_pipeline.py # Model loading and inference
│ └── trained_models/
│ ├── lstm_forecaster.pt # Trained LSTM model
│ └── config.json # Input/output config
│
├── requirements.txt # Python dependencies
├── Dockerfile # Docker config
├── README.md # Project overview
└── .gitignore

## ✅ Requirements

Install dependencies locally (for testing):

```bash
pip install -r requirements.txt

## Run the app locally:

uvicorn app.main:app --reload

## 📁 Example CSV Format

Sample input file (for upload):

0.1,0.2,0.3,0.4,0.5
0.5,0.6,0.7,0.8,0.9


## 🐳 Docker Deployment

FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]


## Build and Run Locally

docker build -t multimodal-forecast .
docker run -p 7860:7860 multimodal-forecast
