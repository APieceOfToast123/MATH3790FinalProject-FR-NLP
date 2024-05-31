import torch
from torch.utils.data import Dataset

def load_data(filepath='poems.txt'):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize_input_data(text):
    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    text_as_int = [char_to_idx[char] for char in text]
    return text_as_int, char_to_idx, idx_to_char

class TextDataset(Dataset):
    def __init__(self, text_as_int, seq_length):
        self.text_as_int = text_as_int
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text_as_int) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        input_seq = torch.tensor(self.text_as_int[start:end-1])
        target_seq = torch.tensor(self.text_as_int[start+1:end])
        return input_seq, target_seq
