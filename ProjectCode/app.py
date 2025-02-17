import streamlit as st
import os
from utils.vocabulary import Vocabulary

from utils.nlp_utils import TextEmbedder
from utils.reinforcement_learning import TextSequenceEnvironment, DQNAgent
import os
from models.model_loader import load_models
from utils.file_processing import process_uploaded_files
from utils.captioning import generate_captions
import traceback

st.set_page_config(
    page_title="RNN Story Generator",
    page_icon="📖",
)

session_state = st.session_state


if "captions" not in session_state:
    session_state.captions = []

if "optimized_sequence" not in session_state:
    session_state.optimized_sequence = []

if "model1_story" not in session_state:
    session_state.model1_story = ""

if "model2_story" not in session_state:
    session_state.model2_story = ""

if "total_images_count" not in session_state:
    session_state.total_images_count = 0

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

        if(session_state.total_images_count > 0 and session_state.total_images_count != len(image_paths)):
            session_state.update(captions=[])
            session_state.update(optimized_sequence=[])
            session_state.update(model1_story="")
            session_state.update(model2_story="")
            session_state.total_images_count = 0
        session_state.total_images_count = len(image_paths)
        print(image_paths)
        
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

                # Create a copy for manipulation
        optimized_captions = session_state.optimized_sequence.copy()
        
        # Display sortable list with manual controls
        new_order = []
        for i, item in enumerate(optimized_captions):
            cols = st.columns([1, 4, 1, 1])
            with cols[0]:
                st.image(item["image_path"], use_container_width=True)
            with cols[1]:
                st.write(item["caption"])
            with cols[2]:
                if st.button("↑", key=f"up_{i}"):
                    if i > 0:
                        optimized_captions[i], optimized_captions[i-1] = optimized_captions[i-1], optimized_captions[i]
                        session_state.optimized_sequence = optimized_captions
                        st.rerun()
            with cols[3]:
                if st.button("↓", key=f"down_{i}"):
                    if i < len(optimized_captions)-1:
                        optimized_captions[i], optimized_captions[i+1] = optimized_captions[i+1], optimized_captions[i]
                        session_state.optimized_sequence = optimized_captions
                        st.rerun()

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
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.text_area("Debug Info:", traceback.format_exc(), height=150)
