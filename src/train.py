import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from .encoder import Encoder
from .decoder import Decoder
from .transformer import Transformer  # the wrapper you defined

# -----------------------------
# Configuration
# -----------------------------
class Config:
    data_path_fi = "data/EUbookshop.fi"
    data_path_en = "data/EUbookshop.en"
    seq_len = 64
    embed_dim = 512
    num_blocks = 6
    heads = 8
    expansion_factor = 4
    dropout = 0.2
    pad_token_id = 0
    batch_size = 32
    lr = 3e-4
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_split = 0.9   # use 10% for testing


# -----------------------------
# Simple Tokenizer
# -----------------------------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, sentences, min_freq=2):
        word_freq = {}
        for sent in sentences:
            for word in sent.strip().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence, max_len):
        tokens = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence.strip().split()]
        tokens = [self.word2idx["<sos>"]] + tokens[: max_len - 2] + [self.word2idx["<eos>"]]
        tokens += [self.word2idx["<pad>"]] * (max_len - len(tokens))
        return torch.tensor(tokens)

    def decode(self, tokens):
        return " ".join([self.idx2word.get(int(t), "<unk>") for t in tokens if t not in [0, 1, 2]])


# -----------------------------
# Dataset
# -----------------------------
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_tokenizer, trg_tokenizer, seq_len):
        self.src = src_sentences
        self.trg = trg_sentences
        self.src_tok = src_tokenizer
        self.trg_tok = trg_tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_seq = self.src_tok.encode(self.src[idx], self.seq_len)
        trg_seq = self.trg_tok.encode(self.trg[idx], self.seq_len)
        return src_seq, trg_seq


# -----------------------------
# Training Loop
# -----------------------------
def train():
    cfg = Config()

    # 1. Load dataset
    with open(cfg.data_path_fi, "r", encoding="utf-8") as f:
        src_sentences = f.readlines()
    with open(cfg.data_path_en, "r", encoding="utf-8") as f:
        trg_sentences = f.readlines()

    # Reduce dataset for faster testing (10%)
    dataset_size = int(0.1 * len(src_sentences))
    src_sentences = src_sentences[:dataset_size]
    trg_sentences = trg_sentences[:dataset_size]

    # Build vocab
    src_tokenizer = SimpleTokenizer()
    trg_tokenizer = SimpleTokenizer()
    src_tokenizer.build_vocab(src_sentences)
    trg_tokenizer.build_vocab(trg_sentences)

    # Create dataset
    dataset = TranslationDataset(src_sentences, trg_sentences, src_tokenizer, trg_tokenizer, cfg.seq_len)

    # Train/test split
    train_size = int(cfg.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # 2. Define model
    model = Transformer(embed_dim=cfg.embed_dim,
                        src_vocab_size=len(src_tokenizer.word2idx),
                        target_vocab_size=len(trg_tokenizer.word2idx),
                        seq_len=cfg.seq_len,
                        num_blocks=cfg.num_blocks,
                        expansion_factor=cfg.expansion_factor,
                        heads=cfg.heads,
                        dropout=cfg.dropout,
                        pad_token_id=cfg.pad_token_id).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # 3. Training
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(cfg.device), trg.to(cfg.device)

            # Shift target for teacher forcing
            trg_input = trg[:, :-1]
            trg_labels = trg[:, 1:]

            # Forward
            outputs = model(src, trg_input)

            # Reshape for loss
            outputs = outputs.reshape(-1, outputs.shape[-1])
            trg_labels = trg_labels.reshape(-1)

            loss = criterion(outputs, trg_labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.epochs}, Training Loss: {avg_loss:.4f}")

    print("✅ Training complete")


if __name__ == "__main__":
    train()
