import streamlit as st
import os
import torch
import tempfile
from zipfile import ZipFile
from PIL import Image
from torchvision import transforms
from helper.model import EncoderCNN, DecoderRNN
# from helper.nlp_utils import clean_sentence
import pickle
from helper.vocabulary import Vocabulary
from transformers import AutoTokenizer, AutoModel
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import boto3

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
# print("Torch version:", torch.__version__)
# print("Streamlit version:", st.__version__)
# print("Transformers version:", transformers.__version__)

# Function to download files from S3
def download_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_key, local_path)

# S3 bucket details
bucket_name = "story-generate-rnn"
vocab_s3_key = "data/vocab.pkl"
encoder_s3_key = "models/encoder.pkl"
decoder_s3_key = "models/decoder.pkl"

# Local paths
vocab_local_path = "./data/vocab.pkl"
encoder_local_path = "./models/encoder.pkl"
decoder_local_path = "./models/decoder.pkl"

# Download files from S3 if they do not exist locally
if not os.path.exists(vocab_local_path):
    download_from_s3(bucket_name, vocab_s3_key, vocab_local_path)

if not os.path.exists(encoder_local_path):
    download_from_s3(bucket_name, encoder_s3_key, encoder_local_path)

if not os.path.exists(decoder_local_path):
    download_from_s3(bucket_name, decoder_s3_key, decoder_local_path)

# Initialize encoder and decoder models
embed_size = 256
hidden_size = 512

# Load vocabulary
with open(vocab_local_path, "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

encoder.load_state_dict(torch.load(encoder_local_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_local_path, map_location=device))

encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()

# Define image preprocessing
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence

# Function to generate captions
def generate_caption(image_path):
    test_image = Image.open(image_path).convert("RGB")
    test_image_tensor = transform_test(test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(test_image_tensor).unsqueeze(1)
        output = decoder.sample(features)

    caption = clean_sentence(output, vocab.idx2word)
    return caption

# Text embedding utility using BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# TextSequenceEnvironment and DQN Agent Classes
class TextSequenceEnvironment:
    def __init__(self, sequences):
        self.sequences = sequences
        self.n = len(sequences)
        self.current_state = list(range(self.n))

    def reset(self):
        random.shuffle(self.current_state)
        return self.current_state

    def step(self, action):
        i, j = action
        self.current_state[i], self.current_state[j] = self.current_state[j], self.current_state[i]
        reward = self._evaluate_sequence()
        done = self._is_terminal()
        return self.current_state, reward, done

    def _evaluate_sequence(self):
        embeddings = [get_text_embedding(self.sequences[i]['caption']) for i in self.current_state]
        reward = sum(
            np.dot(embeddings[i], embeddings[i + 1].T) for i in range(len(embeddings) - 1)
        )
        return reward

    def _is_terminal(self):
        return len(self.current_state) == self.n

    def valid_actions(self):
        return [(i, j) for i in range(len(self.current_state)) for j in range(i + 1, len(self.current_state))]

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.memory = []
        self.batch_size = 32
        self.q_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                target += self.gamma * torch.max(self.q_network(next_state_tensor)).item()
            state_tensor = torch.tensor(state, dtype=torch.float32)
            target_f = self.q_network(state_tensor).detach()
            target = torch.tensor(target, dtype=torch.float32)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.q_network(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Streamlit app
st.title("Image Captioning and Sequence Optimization")

# Upload images
uploaded_files = st.file_uploader("Upload images or zip file", accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tempdir:
        image_paths = []
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                with ZipFile(file, "r") as zip_ref:
                    zip_ref.extractall(tempdir)
                    image_paths.extend(
                        [os.path.join(tempdir, f) for f in zip_ref.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    )
            else:
                path = os.path.join(tempdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getvalue())
                image_paths.append(path)

        # Display images and captions
        if image_paths:
            st.subheader("Uploaded Images with Captions")
            cols = st.columns(3)  # 3-column grid layout
            caption_placeholders = []
            captions = []  # To store captions for each image

            # Display images with "Processing..." as placeholders
            for i, image_path in enumerate(image_paths):
                col = cols[i % 3]
                col.image(image_path, use_container_width=True)
                placeholder = col.empty()
                placeholder.text("Processing...")
                caption_placeholders.append(placeholder)

            # Generate captions and update placeholders
            for i, image_path in enumerate(image_paths):
                caption = generate_caption(image_path)
                captions.append({"image_path": image_path, "caption": caption})
                caption_placeholders[i].text(caption)

            # Optimize sequence
            env = TextSequenceEnvironment(captions)
            state_size = len(captions)
            action_size = len(env.valid_actions())
            agent = DQNAgent(state_size, action_size)

            st.write("Training sequence optimizer...")
            for episode in range(100):  # Fewer episodes for demo
                state = env.reset()
                done = False
                while not done:
                    action = agent.act(state)
                    next_state, reward, done = env.step(env.valid_actions()[action])
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                agent.replay()

            # Generate optimized order
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, _, done = env.step(env.valid_actions()[action])

            optimized_captions = [captions[i] for i in state]

            st.subheader("Optimized Image Sequence")
            cols = st.columns(3)  # 3-column grid for reordered images
            for idx, caption_item in enumerate(optimized_captions):
                col = cols[idx % 3]
                col.image(caption_item["image_path"], caption=caption_item["caption"], use_container_width=True)