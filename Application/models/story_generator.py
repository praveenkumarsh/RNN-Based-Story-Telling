from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

class StoryGenerator:
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, keywords, max_length=300):
        prompt = f"Keywords: {keywords}\nStory:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        story = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return re.search(r'Story:(.*)', story, re.DOTALL).group(1).strip()

class NewStoryGenerator():
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, keywords, max_length=300):
        prompt = f"<|keywords|>{keywords}<|story|>"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.9,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True
        )
        story = self.tokenizer.decode(output[0], skip_special_tokens=False)
        return story.split("<|story|>")[1].replace("<|endoftext|>", "").strip()