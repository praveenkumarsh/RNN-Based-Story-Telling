import streamlit as st
import tempfile
import os
from utils.vocabulary import Vocabulary
import re

from utils.nlp_utils import TextEmbedder
from utils.reinforcement_learning import TextSequenceEnvironment, DQNAgent
import os
import warnings
from models.model_loader import load_models
from utils.file_processing import process_uploaded_files
from utils.captioning import generate_captions

session_state = st.session_state

if "captions" not in session_state:
    session_state.captions = []

if "optimized_sequence" not in session_state:
    session_state.optimized_sequence = []

if "model1_story" not in session_state:
    session_state.model1_story = ""

if "model2_story" not in session_state:
    session_state.model2_story = ""

UPLOAD_DIR = "uploaded_images/"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

def main():
    global session_state
    st.title("RNN Based Story Generation")
    encoder, decoder, vocab, story_gen1, story_gen2 = load_models()
    text_embedder = TextEmbedder()
    
    uploaded_files = st.file_uploader(
        "Upload images or zip file",
        type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        image_paths = process_uploaded_files(uploaded_files, UPLOAD_DIR)
        
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
                col.image(f.read(), use_container_width=True)
            placeholder = col.empty()
            placeholder.text("Processing...")
            caption_placeholders.append(placeholder)
        
        # Generate captions
        if not session_state.captions:
            captions = generate_captions(encoder, decoder, vocab, image_paths)
            session_state.update(captions=captions)
        captions = session_state.captions
        
        # Update placeholders with captions
        for i, caption in enumerate(captions):
            caption_placeholders[i].text(caption["caption"])
        
        # Sequence optimization
        st.write("Optimizing sequence...")

        if not session_state.optimized_sequence:
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
            session_state.update(optimized_sequence=optimized_captions)
        optimized_captions = session_state.optimized_sequence

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

        if st.button("ReOptimize Sequence"):
            session_state.update(optimized_sequence=[]) 
            session_state.update(model1_story="") 
            session_state.update(model2_story="") 
            st.rerun()

        # Sequence optimization
        st.write("Generating Story...")

        if session_state.model1_story == "":
            model1_story = story_gen1.generate(keywords)
            session_state.update(model1_story=model1_story)
        story1 = session_state.model1_story

        if session_state.model2_story == "":
            model2_story = story_gen2.generate(keywords)  
            session_state.update(model2_story=model2_story)                                                   
        story2 = session_state.model2_story

        st.subheader("Generated Story (Model 1)")
        st.write(story1)
        
        st.subheader("Generated Story (Model 2)")
        st.write(story2)
        
        if st.button("Regenerate Story"):
            session_state.update(model1_story="") 
            session_state.update(model2_story="") 
            st.rerun()

if __name__ == "__main__":
    main()
