import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = {
    "embed_size": 1024,
    "hidden_size": 2048,
    "vocab_threshold": 5,
}
# Add constants
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}