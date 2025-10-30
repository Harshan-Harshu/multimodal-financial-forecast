from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pipeline.inference_pipeline import predict_batch, load_model
import numpy as np
import ast

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

model = load_model()

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, series: str = Form(...)):
    try:
        parsed_input = ast.literal_eval(series)
        input_data = np.array(parsed_input, dtype=np.float32)
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)
        preds = predict_batch(input_data, model)
        return templates.TemplateResponse("index.html", {"request": request, "result": preds})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"❌ Error: {e}"})
