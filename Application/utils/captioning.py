import torch
from PIL import Image
from utils.image_processing import get_image_transform
from utils.nlp_utils import clean_sentence
from models.object_detectors import ObjectDetector, YOLOObjectDetector
from config.settings import DEVICE
import streamlit as st
import torch.nn.functional as F

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

# def generate_captions(encoder, decoder, vocab, image_paths, beam_size=20, max_length=400):
#     transform = get_image_transform()
#     captions = []

#     for path in image_paths:
#         image = Image.open(path).convert("RGB")
#         image = transform(image).unsqueeze(0)  # Add batch dimension
        
#         with torch.no_grad():
#             features = encoder(image)

#         start_token = vocab('<start>')
#         end_token = vocab('<end>')

#         # Initialize beam search
#         sequences = [(torch.tensor([start_token]), 0.0)]  # (sequence tensor, log_prob)
#         completed_sequences = []

#         for _ in range(max_length):
#             all_candidates = []

#             for seq, log_prob in sequences:
#                 if seq[-1].item() == end_token:
#                     completed_sequences.append((seq, log_prob))
#                     continue  # Don't expand finished sequences

#                 with torch.no_grad():
#                     outputs = decoder(features, seq.unsqueeze(0))
#                     log_probs = F.log_softmax(outputs[:, -1, :], dim=-1)  # Log probabilities of next word

#                 # Select top-K words
#                 top_log_probs, top_indices = log_probs.topk(beam_size, dim=-1)

#                 for i in range(beam_size):
#                     next_word = top_indices[0, i].item()
#                     new_log_prob = log_prob + top_log_probs[0, i].item()
#                     new_seq = torch.cat([seq, torch.tensor([next_word])])
#                     all_candidates.append((new_seq, new_log_prob))

#             # Select top-k sequences
#             sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

#             # Stop if we have enough completed sequences
#             if len(completed_sequences) >= beam_size:
#                 break

#         # Choose the best sequence (completed or ongoing)
#         best_seq = max(completed_sequences + sequences, key=lambda x: x[1])[0]
#         caption = [vocab.idx2word[idx.item()] for idx in best_seq if idx.item() != start_token]
        
#         captions.append({"image_path": path, "caption": " ".join(caption)})
    
#     return captions
