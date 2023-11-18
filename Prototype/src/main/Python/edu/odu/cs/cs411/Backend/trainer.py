import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from music21 import *
import os

processed_notesData = []
def process_musicXMLData(musicscore):
    
    
    
    
    return processed_notesData

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