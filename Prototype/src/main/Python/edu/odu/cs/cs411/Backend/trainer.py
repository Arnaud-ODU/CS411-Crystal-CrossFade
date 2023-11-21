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

root_directory = ""
batch_size = 32

dataset = MusicXMLDataset(root_directory)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)