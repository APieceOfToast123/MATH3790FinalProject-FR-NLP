import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_helpers import load_data, tokenize_input_data, TextDataset
from model import TransformerModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train(dataset, vocab_size, model_dir='./model_save/model.pt', num_epochs=20):
    model = TransformerModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (input_seq, target_seq) in enumerate(dataset):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad()
            output = model(input_seq, input_seq)
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")
        avg_loss = total_loss / len(dataset)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        scheduler.step()

    torch.save(model.state_dict(), model_dir)

if __name__ == '__main__':
    text = load_data('poems.txt')
    text_as_int, char_to_idx, idx_to_char = tokenize_input_data(text)

    seq_length = 100
    dataset = TextDataset(text_as_int, seq_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    vocab_size = len(char_to_idx)
    train(dataloader, vocab_size)
