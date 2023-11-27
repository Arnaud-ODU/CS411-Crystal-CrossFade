import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from music21 import *
import os
from torch.utils.data import Dataset, DataLoader


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
    note_features = []
    for note in score.flat.notes:
        # Extract note features
        if (note.isNote):
             note_features = [note.measureNumber, note.pitch.midi, note.duration.quarterLength]
    
        if (note.isChord):
            for x in note._notes:
             note_features = [x.measureNumber, x.pitch.midi, x.duration.quarterLength]
       # note_features = [
        #    note.pitch.midi,  # MIDI number for pitch
         #   note.duration.quarterLength
        #]
        note_features.extend(parse_beams(note.beams))
        input_sequence.append(note_features)

        current_beam_sequence = parse_beams(note.beams)
        current_target_sequence = [encode_beam(b) for b in current_beam_sequence]
        target_sequence.append(current_target_sequence)

        # Convert lists to PyTorch tensors
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32)
    #print(input_tensor)
    target_tensor = torch.tensor(target_sequence, dtype=torch.long)
    #print(target_tensor)
    return input_tensor, target_tensor

class MusicXMLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#RNN Model for padding uneven sequence lengths for tensors    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
       super(RNNModel, self).__init__()
       self.embedding = nn.Embedding(input_size, hidden_size)
       self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
       self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
       embedded = self.embedding(x)
       packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
       out, _ = self.rnn(packed)
       unpacked, _ = pad_packed_sequence(out, batch_first=True)
       out = self.linear(unpacked[:, -1, :])
       return out

input_size =  7 # 7 Features(Measure Number,Pitch.midi, note duration in int, [Beam Start, Beam Continue, Beam Stop, Beam Partial(Left/Right)])
hidden_size = 64
output_size = 4 
batch_size = 120
num_epochs = 100
model = RNNModel(input_size, hidden_size, output_size)
music_dataset = MusicXMLDataset(parse_musicxml('Control_Tests'))
dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, data_loader):
    for epoch in range(num_epochs):
        for batch in data_loader:
         optimizer.zero_grad()
         sequences, lengths = zip(*batch)

         x_batch = pad_sequence([torch.tensor(seq) for seq in sequences], batch_first=True)
         lengths = torch.tensor(lengths)
         y_batch = torch.randn((x_batch.size(0), output_size))
         
         outputs = model(x_batch, lengths)
         loss = criterion(outputs, y_batch)
         loss.backward()
         optimizer.step()
         
         if (batch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch+1}/{len(dataloader)}], Loss: {loss.item()}')

train_model(model, dataloader)
torch.save(model, 'music_note_model.pth')

print('Training complete')