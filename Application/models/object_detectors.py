import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class ObjectDetector:
    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval()
        self.transform = self.weights.transforms()
    
    def detect_objects(self, image_path, confidence_threshold=0.7):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        results = [
            self.weights.meta["categories"][i]
            for i, score in zip(predictions["labels"], predictions["scores"])
            if score >= confidence_threshold
        ]
        
        return results

class YOLOObjectDetector:
    def __init__(self, model_size='yolov8n.pt'):
        self.model = YOLO(model_size)
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        results = self.model(image_path, conf=confidence_threshold)

        detections = []
        for result in results:
            class_ids = result.boxes.cls.tolist()
            detections.extend([self.model.names[int(cls_id)] for cls_id in class_ids])
        
        return detections
