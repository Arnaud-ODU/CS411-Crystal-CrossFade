import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from music21 import *
import os

scorenote = []
def process_musicXMLData(file_path):
    score = converter.parse(file_path)
    # Extract relevant information from the parsed MusicXML file
    # Return the necessary data (e.g., notes, durations, etc.)
    scorenote = []
    for note in score.flatten().notes:
       #Check if note is a note or chord(A collection of 3 notes or more being played at the same time)
        if (note.isNote):
             scorestring = str(note.measureNumber)+ ", "+ str(note.pitch) + ", " + str(note.duration.type) + ", " + str(note.duration.quarterLength)+", " +str(note.beams);
             scorenote.append(scorestring)
    
        if (note.isChord):
            for x in note._notes:
             scorestring = str(x.measureNumber)+ ", "+ str(x.pitch) + ", " + str(x.duration.type) + ", " + str(x.duration.quarterLength)+", " +str(x.beams);
             scorenote.append(scorestring)

    return scorenote


class MusicXMLDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(".xml")]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        score = converter.parse(file_path)
        processed_notesData = process_musicXMLData(score)
        return processed_notesData




data_directory = "Data"
batch_size = 64



class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Consider only the last output in the sequence
        return out

# Define the model parameters
input_size = 3  # Number of features
hidden_size = 64
num_layers = 2
output_size = 1  # Example output size (adjust based on task)

train_dataset = MusicXMLDataset(data_directory)
train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = RNNModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for data in train_dataLoader:
        optimizer.zero_grad()
        inputs = data  # Assuming no separate labels, input and output are the same
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Example loss calculation (input=output in this case)
        loss.backward()
        optimizer.step()
        
        if (data+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataLoader)}], Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'music_note_model.ckpt')

print('Training complete')
