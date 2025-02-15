import os

# Get the parent directory of the current file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "vocab": os.path.join(BASE_DIR, "storydata/vocab.pkl"),
    "encoder": os.path.join(BASE_DIR, "storymodels/encoder.pkl"),
    "decoder": os.path.join(BASE_DIR, "storymodels/decoder.pkl"),
    "story_model": os.path.join(BASE_DIR, "storymodels/fine_tuned_gpt2"),
    "story_model_new": os.path.join(BASE_DIR, "storymodels/fine_tuned_gpt2_new"),
}