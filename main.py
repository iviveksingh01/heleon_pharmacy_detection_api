from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
from collections import Counter
from datetime import datetime

app = FastAPI(title="Haleon Object Detection API")

# Load your trained model
model = YOLO("best (1).pt")  # Make sure this file exists in the same folder

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        
        results = model(image)
        
        class_counts = Counter()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                class_counts[class_name] += 1
        
        total_count = sum(class_counts.values())

        # Calculate share for each class
        class_shares = {}
        sensodyne_total = 0  # Sum of all Sensodyne products

        for class_name, count in class_counts.items():
            share = (count / total_count) * 100 if total_count > 0 else 0.0
            class_shares[class_name] = {
                "count": count,
                "visibility of share": round(share, 2)
            }
            # Check if class name starts with "SENSODYNE"
            if class_name.startswith("SENSODYNE"):
                sensodyne_total += count

        # Calculate overall Sensodyne share
        share_of_sensodyne = (sensodyne_total / total_count) * 100 if total_count > 0 else 0.0

        response = {
            "result": class_shares,
            "total_count": total_count,
            "width": width,
            "height": height,
            "input_image": file.filename,
            "latest_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "haleon_best_v1",
            "store_id": "a",
            "visibility_percentage_of_sensodyne_product": round(share_of_sensodyne, 2)  # New key!
        }
        return response

    except Exception as e:
        return {"error": str(e)}