from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the predict function from the src folder
from predict import predict

app = FastAPI()

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    image = Image.open(io.BytesIO(await file.read()))
    
    # Use the predict function to predict the emotion
    label = predict(image)
    
    # Return the predicted emotion as a JSON response
    return {"emotion": label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
