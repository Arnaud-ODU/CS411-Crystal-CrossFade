import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from music21 import converter, beam

def encode_beam(beam):
    if beam == '1beam':
        return 1
    elif beam == '2beam':
        return 2
    elif beam == '3/partial/left':
        return 3
    else:
        return 0  # Default case, for unknown or no beam

def parse_beams(beam_data):
    beam_list = [0,0,0,0]
    for b in beam_data.beamsList:
        if b.type == 'start':
            beam_list = [1,0,0,0]
        elif b.type == 'continue':
            beam_list = [0,2,0,0]
        elif b.type == 'stop':
            beam_list = [0,0,3]
        else:  # For partial or unknown types
            beam_list.append(4)
    return beam_list

def parse_musicxml(file_path):
    # Parsing musicXML file to extract features
    score = converter.parse(file_path)
    input_sequence = []
    target_sequence = []

    for note in score.flat.notes:
        # Extract note features
        note_features = [
            note.pitch.midi,  # MIDI number for pitch
            note.duration.quarterLength
        ]
        note_features.extend(parse_beams(note.beams))
        input_sequence.append(note_features)

        current_beam_sequence = parse_beams(note.beams)
        current_target_sequence = [encode_beam(b) for b in current_beam_sequence]
        target_sequence.append(current_target_sequence)

        # Convert lists to PyTorch tensors
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)
    return input_tensor, target_tensor

class MusicXMLDataset(Dataset):
    def __init__(self, data_folder):
        self.file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return parse_musicxml(file_path)

# Define a simple RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # We're interested in the last output for this sequence
        return out

# Define the model parameters
input_size = 6  # 3 for note attributes + 3 for beam encoding
hidden_size = 64
output_size = 10 
num_layers = 1
learning_rate = 1e-3
batch_size = 16
num_epochs = 100

# Load wer dataset
dataset = MusicXMLDataset(data_folder='Control_Tests')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()  # Assuming classification task for simplicity
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # Ensure data is in the correct format for the model
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

# Save the model
torch.save(model, 'music_note_model.pth')

print('Training complete')
