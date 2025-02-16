import os
import streamlit as st
from config.settings import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS

def process_uploaded_files(uploaded_files, temp_dir):
    image_paths = []
    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File {file.name} exceeds {MAX_FILE_SIZE_MB}MB")
            continue
        
        extension = file.name.split(".")[-1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            st.error(f"Unsupported file format: {file.name}")
            continue
        
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        image_paths.append(path)

    return image_paths
