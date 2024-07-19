import torch
import cv2
import re
import nltk
import numpy as np
from nltk import tokenize
from collections import Counter

# configs
img_size = 200 if torch.cuda.is_available() else 128


def read_image(img):
    img_array = np.array(img)
    img = cv2.resize(img_array, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def tokenize_caption(cap_text):
    return tokenize.word_tokenize(str(cap_text).lower())


def process_text(text):
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text_tokens = tokenize_caption(text)
    tokens = [t for t in text_tokens if t not in nltk.corpus.stopwords.words("english")]

    return tokens


def create_vocabulary(text_dataset):
    all_tokens = []
    for text in text_dataset:
        tokens = process_text(text)
        all_tokens.extend(tokens)

    vocab_counter = Counter(all_tokens)
    vocab = [word for word, count in vocab_counter.most_common() if count >= 1]
    vocab = ["<pad>", "<start>", "<end>", "<unk>"] + vocab

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}

    return word_to_idx, idx_to_word
