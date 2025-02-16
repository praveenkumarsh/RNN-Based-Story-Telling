import streamlit as st
import tempfile
import os
import pickle
import re
from zipfile import ZipFile
import torch
from PIL import Image
import numpy as np

from config.paths import PATHS
from config.settings import DEVICE, MODEL_PARAMS
from models.encoder_decoder import EncoderCNN, DecoderRNN
from models.story_generator import StoryGenerator, NewStoryGenerator
from utils.vocabulary import Vocabulary
from utils.image_processing import get_image_transform
from utils.nlp_utils import clean_sentence, TextEmbedder
from utils.reinforcement_learning import TextSequenceEnvironment, DQNAgent
import os
import warnings
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

warnings.filterwarnings("ignore", category=UserWarning, module="torch._classes")

# Add constants
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

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
        
        labels = [self.weights.meta["categories"][i] for i in predictions["labels"]]
        scores = predictions["scores"].tolist()
        boxes = predictions["boxes"].tolist()
        
        results = []
        detected_objects = []
        for label, score, box in zip(labels, scores, boxes):
            if score >= confidence_threshold:
                results.append({
                    "label": label,
                    "score": score,
                    "box": box
                })
                detected_objects.append(label)
        
        return detected_objects
    
class YOLOObjectDetector:
    def __init__(self, model_size='yolov8n.pt'):
        self.model = YOLO(model_size)
        self.class_names = self.model.names
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        results = self.model(image_path, conf=confidence_threshold)
        
        detections = []
        detection_objects = []
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            confidences = result.boxes.conf.tolist()
            class_ids = result.boxes.cls.tolist()
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    "label": self.class_names[int(cls_id)],
                    "score": conf,
                    "box": box
                })
                detection_objects.append(self.class_names[int(cls_id)])
        
        return detection_objects

@st.cache_resource
def load_models():
    with open(PATHS["vocab"], "rb") as f:
        vocab = pickle.load(f)
    
    encoder = EncoderCNN(MODEL_PARAMS["embed_size"])
    decoder = DecoderRNN(
        MODEL_PARAMS["embed_size"],
        MODEL_PARAMS["hidden_size"],
        len(vocab))
    
    encoder.load_state_dict(torch.load(PATHS["encoder"], map_location=DEVICE))
    decoder.load_state_dict(torch.load(PATHS["decoder"], map_location=DEVICE))
    
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    encoder.eval()
    decoder.eval()
    
    story_gen1 = StoryGenerator(PATHS["story_model"])
    story_gen2 = NewStoryGenerator(PATHS["story_model_new"])
    
    object_detector = ObjectDetector()
    return encoder, decoder, vocab, story_gen1, story_gen2, object_detector

def process_uploaded_files(uploaded_files, temp_dir):
    image_paths = []
    for file in uploaded_files:
        # File size validation
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File {file.name} exceeds size limit of {MAX_FILE_SIZE_MB}MB")
            continue
        
        # File type validation
        extension = file.name.split(".")[-1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            if extension == "zip":
                continue  # Handle zip separately
            st.error(f"Unsupported file format: {file.name}. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")
            continue
        
        # Save valid files
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(path)
    
    return image_paths

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
            captions.append({"image_path": path, "caption": caption +" {"+ " ".join(detected_objects) +"}"})
        except Exception as e:
            st.error(f"Error processing {path}: {str(e)}")
    return captions

def main():
    st.title("RNN Based Story Generation")
    encoder, decoder, vocab, story_gen1, story_gen2, object_detector = load_models()
    text_embedder = TextEmbedder()
    
    uploaded_files = st.file_uploader(
        "Upload images or zip file",
        type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = process_uploaded_files(uploaded_files, temp_dir)
            
            if not image_paths:
                st.warning("No valid images found in upload")
                return
                
            # Display images and generate captions
            st.subheader("Uploaded Images with Captions")
            cols = st.columns(3)
            caption_placeholders = []
            
            # First pass: Show images with loading placeholders
            for i, path in enumerate(image_paths):
                col = cols[i % 3]
                with open(path, "rb") as f:
                    col.image(f.read(), use_container_width =True)
                placeholder = col.empty()
                placeholder.text("Processing...")
                caption_placeholders.append(placeholder)
            
            # Generate captions
            captions = generate_captions(encoder, decoder, vocab, image_paths)
            
            # Update placeholders with captions
            for i, caption in enumerate(captions):
                caption_placeholders[i].text(caption["caption"])
            
            # Sequence optimization
            st.write("Optimizing sequence...")
            env = TextSequenceEnvironment(captions, text_embedder)
            state_size = len(captions)
            action_size = len(env.valid_actions())
            agent = DQNAgent(state_size, action_size)
            
            # Training loop
            for episode in range(50):  # Reduced for demo
                state = env.reset()
                done = False
                while not done:
                    action_idx = agent.act(state)
                    action = env.valid_actions()[action_idx]
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action_idx, reward, next_state, done)
                    state = next_state
                agent.replay()
            
            # Generate optimized sequence
            state = env.reset()
            done = False
            while not done:
                action_idx = agent.act(state)
                action = env.valid_actions()[action_idx]
                state, _, done = env.step(action)
            
            optimized_captions = [captions[i] for i in state]
            
            # Display optimized sequence
            st.subheader("Optimized Image Sequence")
            cols = st.columns(3)
            for idx, item in enumerate(optimized_captions):
                col = cols[idx % 3]
                col.image(item["image_path"], 
                         caption=item["caption"], 
                         use_container_width=True)
            
            # Generate stories
            caption_texts = [item["caption"] for item in optimized_captions]
            keywords = ", ".join(caption_texts)

            # Story coherence check
            # coherence_score = calculate_coherence(caption_texts)
            # if coherence_score < 0.3:
            #     st.warning("⚠️ The generated sequence has low coherence. Story might be disjointed.")
                        
            st.subheader("Generated Story (Model 1)")
            story1 = story_gen1.generate(keywords)
            story_text1 = re.search(r'Story:(.*)', story1, re.DOTALL)
            st.write(story_text1.group(1).strip() if story_text1 else story1)
            
            st.subheader("Generated Story (Model 2)")
            story2 = story_gen2.generate(keywords)
            st.write(story2)

if __name__ == "__main__":
    main()