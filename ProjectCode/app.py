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

def main():
    st.title("RNN Based Story Generation")
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
                    col.image(f.read(), use_container_width=True)
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