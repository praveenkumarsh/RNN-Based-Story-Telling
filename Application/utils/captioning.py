import torch
from PIL import Image
from utils.image_processing import get_image_transform
from utils.nlp_utils import clean_sentence
from models.object_detectors import ObjectDetector, YOLOObjectDetector
from config.settings import DEVICE
import streamlit as st

def generate_captions(encoder, decoder, vocab, image_paths):
    transform = get_image_transform()
    captions = []

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                features = encoder(image_tensor).unsqueeze(1)
                output = decoder.sample(features)

            caption = clean_sentence(output, vocab.idx2word)

            object_detector = ObjectDetector()
            detections = object_detector.detect_objects(path)

            yolo_detector = YOLOObjectDetector()
            yolo_detections = yolo_detector.detect_objects(path)

            detected_objects = list(set(detections + yolo_detections))
            captions.append({
                "image_path": path,
                "caption": caption + " {" + " ".join(detected_objects) + "}"
            })
        except Exception as e:
            st.error(f"Error processing {path}: {str(e)}")
    
    return captions
