import os
import pickle
from collections import Counter
import nltk
from pycocotools.coco import COCO

class Vocabulary:
    def __init__(self, vocab_threshold, vocab_file, annotations_file):
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.annotations_file = annotations_file
        self.start_word = "<start>"
        self.end_word = "<end>"
        self.unk_word = "<unk>"
        self.idx2word = {}
        self.word2idx = {}
        self.idx = 0
        self._initialize()

    def _initialize(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
            self.__dict__ = vocab.__dict__
        else:
            self._build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def _build_vocab(self):
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self._add_captions()

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def _add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, idx in enumerate(ids):
            caption = str(coco.anns[idx]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        for word in words:
            self.add_word(word)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_word])

    def __len__(self):
        return len(self.word2idx)