import pickle
import torch
from config.paths import PATHS
from config.settings import DEVICE, MODEL_PARAMS
from models.encoder_decoder import EncoderCNN, DecoderRNN
from models.story_generator import StoryGenerator, NewStoryGenerator
from models.object_detectors import ObjectDetector
import streamlit as st

@st.cache_resource
def load_models():
    with open(PATHS["vocab"], "rb") as f:
        vocab = pickle.load(f)
    
    encoder = EncoderCNN(MODEL_PARAMS["embed_size"])
    decoder = DecoderRNN(
        MODEL_PARAMS["embed_size"],
        MODEL_PARAMS["hidden_size"],
        len(vocab)
    )
    
    encoder.load_state_dict(torch.load(PATHS["encoder"], map_location=DEVICE))
    decoder.load_state_dict(torch.load(PATHS["decoder"], map_location=DEVICE))

    encoder.to(DEVICE)
    decoder.to(DEVICE)
    encoder.eval()
    decoder.eval()

    story_gen1 = StoryGenerator(PATHS["story_model"])
    story_gen2 = NewStoryGenerator(PATHS["story_model_new"])

    return encoder, decoder, vocab, story_gen1, story_gen2