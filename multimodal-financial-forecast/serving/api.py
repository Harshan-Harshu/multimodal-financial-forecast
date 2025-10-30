from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import uvicorn

from pipeline.inference_pipeline import load_model, predict_batch

app = FastAPI()
templates = Jinja2Templates(directory="serving/templates")
model = load_model()

class TimeSeriesBatch(BaseModel):
    data: List[List[float]]

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, series: str = Form(...)):
    try:
        batch = eval(series)  # Replace this with safe parsing for production
        result = predict_batch(batch, model)
        return templates.TemplateResponse("index.html", {"request": request, "result": result})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {str(e)}"})

@app.post("/predict")
async def predict_api(input: TimeSeriesBatch):
    return {"prediction": predict_batch(input.data, model)}
