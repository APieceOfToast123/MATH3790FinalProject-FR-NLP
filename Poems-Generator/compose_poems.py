import torch
from model import TransformerModel
from data_helpers import load_data, tokenize_input_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_text(model, start_string, int_to_word, word_to_int, num_generate=100):
    model.eval()
    input_eval = torch.tensor([word_to_int[s] for s in start_string], dtype=torch.long).unsqueeze(0).to(device)
    text_generated = []

    with torch.no_grad():
        for _ in range(num_generate):
            output = model(input_eval, input_eval)
            predictions = output[:, -1, :]
            predicted_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1).item()
            input_eval = torch.cat([input_eval, torch.tensor([[predicted_id]], device=device)], dim=1)
            text_generated.append(int_to_word[predicted_id])

    return start_string + ''.join(text_generated)

if __name__ == '__main__':
    text = load_data('poems.txt')
    text_as_int, char_to_idx, idx_to_char = tokenize_input_data(text)

    model = TransformerModel(len(char_to_idx)).to(device)
    model.load_state_dict(torch.load('model_save/model.pt'))
    
    start_string = '机器学习'
    print(generate_text(model, start_string, idx_to_char, char_to_idx))


// ///////////////////////















