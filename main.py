from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import io
from PIL import Image
from collections import Counter
import json
from datetime import datetime


app = FastAPI(title="Haleon Object Detection API")


model = YOLO("best.pt")  

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        
        width, height = image.size
        
        # Run inference
        results = model(image)
        
        # Count detections per class
        class_counts = Counter()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                class_counts[class_name] += 1
        
        # Calculate total count
        total_count = sum(class_counts.values())
        
        # Calculate share (%) for each class
        class_shares = {}
        for class_name, count in class_counts.items():
            share = (count / total_count) * 100 if total_count > 0 else 0.0
            class_shares[class_name] = {
                "count": count,
                "share": round(share, 2)  # Rounded to 2 decimal places
            }
        
        # Build final response
        response = {
            "result": class_shares,  # Now includes count + share
            "total_count": total_count,
            "width": width,
            "height": height,
            "input_image": file.filename,  # Name of uploaded file
            "latest_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "haleon_best_v1",
            "store_id": "a"
        }
        
        return response

    except Exception as e:
        return {"error": str(e)}