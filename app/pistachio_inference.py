# Import libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from pistachio.src.models.lightning_predict import *
import os


# Replace 'your_model_folder' with the actual path to your model folder
model_file = "app/models/transfer_learning_model.pth"

# Create FastAPI app
app = FastAPI()

# Define a route for image classification
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Ensure the 'temp' folder exists
        temp_folder = "temp"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # Construct the image path
        image_path = os.path.join(temp_folder, file.filename)
        print(f"Image path: {image_path}")

        # Write the image
        with open(image_path, "wb") as image_file:
            image_file.write(await file.read())

        predicted_class = make_prediction(image_path, model_file)

        # Return the predicted class as JSON
        return {"predicted_class": predicted_class}

    except Exception as e:
        # Handle errors and return an HTTP 500 status code
        raise HTTPException(status_code=500, detail=str(e))

