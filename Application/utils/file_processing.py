import os
import zipfile
import streamlit as st
from config.settings import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS

def process_uploaded_files(uploaded_files, temp_dir):
    image_paths = []
    
    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File {file.name} exceeds {MAX_FILE_SIZE_MB}MB")
            continue

        extension = file.name.split(".")[-1].lower()
        
        if extension in ALLOWED_EXTENSIONS and extension != "zip":
            # Save image file
            path = os.path.join(temp_dir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(path)

        elif extension == "zip":
            # Extract ZIP file
            zip_path = os.path.join(temp_dir, file.name)
            with open(zip_path, "wb") as f:
                f.write(file.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Collect image paths from extracted files
            for root, _, files in os.walk(temp_dir):
                for extracted_file in files:
                    ext = extracted_file.split(".")[-1].lower()
                    if ext in ALLOWED_EXTENSIONS:
                        image_paths.append(os.path.join(root, extracted_file))
        else:
            st.error(f"Unsupported file format: {file.name}")

    return image_paths
