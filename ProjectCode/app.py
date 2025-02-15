import streamlit as st
import tempfile
import os
import pickle
import re
from zipfile import ZipFile
import torch
from PIL import Image

from config.paths import PATHS
from config.settings import DEVICE, MODEL_PARAMS
from models.encoder_decoder import EncoderCNN, DecoderRNN
from models.story_generator import StoryGenerator, NewStoryGenerator
from utils.vocabulary import Vocabulary
from utils.image_processing import get_image_transform
from utils.nlp_utils import clean_sentence, TextEmbedder
from utils.reinforcement_learning import TextSequenceEnvironment, DQNAgent

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
    
    return encoder, decoder, vocab, story_gen1, story_gen2

def process_uploaded_files(uploaded_files, temp_dir):
    image_paths = []
    for file in uploaded_files:
        if file.name.lower().endswith(".zip"):
            with ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
                image_paths.extend([
                    os.path.join(temp_dir, f) 
                    for f in zip_ref.namelist()
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ])
        else:
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
            captions.append({"image_path": path, "caption": caption})
        except Exception as e:
            st.error(f"Error processing {path}: {str(e)}")
    return captions

def main():
    st.title("Image Captioning and Sequence Optimization")
    encoder, decoder, vocab, story_gen1, story_gen2 = load_models()
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
            
            st.subheader("Generated Story (Model 1)")
            story1 = story_gen1.generate(keywords)
            story_text1 = re.search(r'Story:(.*)', story1, re.DOTALL)
            st.write(story_text1.group(1).strip() if story_text1 else story1)
            
            st.subheader("Generated Story (Model 2)")
            story2 = story_gen2.generate(keywords)
            st.write(story2)

if __name__ == "__main__":
    main()