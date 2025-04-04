from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from PIL import Image
import io
from src.predict import predict  # Import the predict function from predict.py

# Initialize the FastAPI app
app = FastAPI()

# Set up Jinja2 template rendering
templates = Jinja2Templates(directory="templates")

# Route to serve the HTML page (form for image upload)
@app.get("/predict/", response_class=HTMLResponse)
async def get_predict_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle image upload and make a prediction
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Get the prediction using the predict function from predict.py
    predicted_class = predict(image)

    # Return the result as a JSON response
    result = {"prediction": predicted_class}
    return JSONResponse(content=result)
