import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PARAMS = {
    "embed_size": 256,
    "hidden_size": 512,
    "vocab_threshold": 5,
}