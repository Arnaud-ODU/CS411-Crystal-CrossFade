# Trainer Class
# This Class Will Be Used To Train With Corrected Beam Folder
# And Use The Evaluation Model to predict Beam Errors From Uploaded MXL,XML,MusicXML
# Files For the User to see.
#
#
# A Trainer Has:
#  Pre-Processing of Data - to be prepared to be used as inputs and labels and turned to tensors
#  DataSet Class - Access to Folders from training and testing folders and applies the transformations from raw data to tensors
#  Neural Network Model - Structure of the machine learning and important for the behaviour and result of your model.
#  Training - Uses the model and trains it with the training dataset
#  Evaluation - Uses the model and tests it with the training dataset
#@author Mohamed Abdullahi, Arnaud
#CS 411W Prof. Kennedy


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


# Parse the beams to be in the format of [Start,Continue,Stop,Partial]
def parse_beams(beam_data):
    beam_list = [0,0,0,0] # Initialize the beam list to 0 for all four dimensions
    for b in beam_data.beamsList:
        if b.type == 'start': # If a beam contains a start set the value of index 0 to 1
            beam_list[0] = 1
        elif b.type == 'continue': # If a beam contains a continue set the value of index 1 to 2
            beam_list[1] = 2
        elif b.type == 'stop': # If a beam contains a stop set the value of index 2 to 3
            beam_list[2] = 3
        else:                   # If a beam contains a partial or backward/forward hook set the value of index 3 to 4
            beam_list[3] = 4
    return beam_list # Return value to be used in parse_musicxml

# Parse the mxl/MusicXML file in the format of [Note/Chord measure number, Note/Chord pitch in integer format, Note/Chord duration, [Beam Info] ]
def parse_musicxml(file_path):
    # Parsing musicXML file to extract features
    score = converter.parse(file_path) # Music21 toolkit to parse mxl/musicXML files
    collection_features = [] # Collection of all note for each note/chord
   
    for note in score.flat.notes:
        # Extract features for each note or chord
        if note.isNote: 
            note_features = [note.measureNumber, note._getMeasureOffset(score), note.pitch.ps, note.duration.quarterLength]
        else: 
            note_features = [note.measureNumber, note._getMeasureOffset(score), sum(p.midi for p in note.pitches) / len(note.pitches), note.duration.quarterLength]
        # Add beam info to the note features for each note/chord
        note_features.extend(parse_beams(note.beams))
        collection_features.append(note_features)    
    
    return collection_features  # Return value to be used in Dataset class

class MusicXMLDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.xml', '.mxl', '.musicxml'))]
        for file in file_paths:  # Access each file in folder to be parsed
            parsed_data = parse_musicxml(file)
            for b in parsed_data:  # Turn the note features into tensors
                self.data.append(torch.tensor([float(feature) for feature in b], dtype=torch.float32))
        self.data = torch.stack(self.data)

    def __len__(self): # Length of list of stacks
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        inputs = data_item[:4] # Use the first four as input [Measure Number, Note Offset, Note pitch, Note Duration]
        # The Beam information will be used as labels for the note information. Labels will be 1 for beam and 0 for no beam.
        if int(data_item[4] > 0):  # If Beam 'Start' is 1, make label as 1.
            label = torch.tensor(int(data_item[4] > 0), dtype=torch.float32)       
        elif int(data_item[5] > 0): # If Beam 'Continue' is 2, make label as 1.
            label = torch.tensor(1, dtype=torch.float32)          
        elif int(data_item[6] > 0): #If Beam 'Stop' is 3, make label as 1
             label = torch.tensor(1, dtype=torch.float32)
        else:                       # Else make label as 0.
            label = torch.tensor(0,dtype=torch.float32)
        #print(inputs, label)                                 
        return inputs, label # Returns Values for trainer

# Note-Beam Prediction Module   
class NBPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers): # Initialization for NBPModel Class
        super(NBPModel, self).__init__() # Initializes nn.Module to inherit the functions
        # Long Short Term Memonry that gives the inputs tensors first dibs on batch dimension
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # Long Short Term Memory Layer
        #Full Connected linear layer that turns the output from the LSTM layer which is the hiddensize into a single output
        self.fc = nn.Linear(hidden_size, 1) # Linear Layer
        # Sigmoid Activation, which creates probabilities from the output 0 or 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # Forward Pass of the Neural network
        x = x.unsqueeze(1) # Takes input tensor X and add an extra dimension to match sequence for LSTM model
        # Output has the hidden states for each LSTM time step, by passing the x tensor through the LSTM Layer, ignoring final hidden state
        out, _ = self.lstm(x) 
        # The output from the last time step in the LSTM is now passed in the connected layer  
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out).squeeze()  # Apply sigmoid activation for binary classification and remove single dimensions
        return out # Return the prediction to be used in the DataLoader and trainer/evalution


# Model Parameters
input_size = 4  # 4 Features(Measure Number,note offset ,Pitch.midi, note duration in int)
hidden_size = 3  # Hidden State Size, Hidden state of each time step of vector length 6(2/3 size of input + outputsize)
batch_size = 32  # Large batch size leads to fast to training but lower accuracy. Should be 16,32,64,128
num_epochs = 4  # Number of times all training data is used once to update parameters. Should be between 1-10
num_layers = 2  # Stack of layers (i.e. LSTM and Linear)
learning_rate = 0.001  # Learning rate for optimizers to update paramaters

# Paths to the training and testing folders
train_folder_path = "~/CS411-Crystal-CrossFade/Prototype/src/main/Python/edu/odu/cs/cs411/Backend/Correct_Beams" # path to mxl files containing correct beams format
#test_folder_path = "~/CS411-Crystal-CrossFade/Prototype/src/main/Python/edu/odu/cs/cs411/Backend/Incorrect_Beams" # path to mxl files containing incorrect beams format

# Instantiate the model and define loss function and optimizer
model = NBPModel(input_size, hidden_size, num_layers) # Neural Network Model to use sequential data and patterns to predict the most likely scenario
criterion = nn.MSELoss() # Mean-Square Error Loss, best for regression. Outputs number P ∈[0,1]
optimizer = torch.optim.Adam(model.parameters(), learning_rate) # Best default optimizer,default learning rate is 0.001, best results 0.002-0.003. If facing overfitting adjust learning rate.

# Create instances of the custom dataset
train_dataset = MusicXMLDataset(train_folder_path)
#test_dataset = MusicXMLDataset(test_folder_path)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Train model using training dataset
def train_model(num_epochs):
        model.train() # Start Training
        for epoch in range(num_epochs): # For Loop of complete iteration of dataset
            runningLoss = 0.0 # Loss in each epoch of batch
            for inputs, labels in train_loader: # For Loop of batches of inputs and labels of the dataset from the dataloader 
                optimizer.zero_grad() # Set gradients of variables to 0.
                outputs = model(inputs) # Predictions from inputs. Raw Logit
                loss = criterion(outputs, labels) # Loss between the outputs and labels.
                loss.backward() # The gradient loss from what the model can learn
                optimizer.step() # Update parameters with gradients from loss.backward()
                runningLoss += loss.item() # Updates running loss value
                #print("Output shape:", outputs.shape)  For debugging
                #print("Labels shape:", labels.shape)  For debugging
            # Once all batches have been iterated through then print each epoch loss.
            epochLoss = runningLoss / len(train_loader.dataset) # Calculates the average loss of each epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epochLoss:.4f}') # Print the Epoch Number and its loss.

torch.save(model, 'music_note_model.pth')
print('Training complete')

train_model(num_epochs)

# Test model using testing dataset, prints the predicted notes with its information that need beams.
def evaluate_beam_predictions(input_test_loader):
        model.eval()  # Turns off Training and starts Evaluation
        beam_predictions = []  # Collection of predicted notes information
        with torch.no_grad():  # Since we are evaluating we do not need gradient calcs, so it gets turned off.
            for inputs, labels in input_test_loader:  # For Loop of batches of inputs and labels of the dataset from the dataloader#
                outputs = model(inputs) # Predictions from inputs in the form of a raw Logit(log( p /(1 - p)))
                predictedP = torch.sigmoid(outputs) # Converts the logits into a probability
                # Classification using a threshold of 0.5, if the prob. is higher than the threshold that means its most likely 
                # a beam and given 1, else 0 for no beam
                predictedL = (predictedP > 0.5).float() 

                if predictedL.sum() > 0:  # Check to see if there are any beam predictions by taking the sum of all label predictions in the batch
                    for i in range(inputs.size(0)):
                        inputData = inputs[i]
                        measure, offset, pitch, duration = inputData[:4].tolist() # Collect first 4 elements
                        beam_predictions.append((measure, offset, pitch, duration)) # Add collected notes data to the collecetion of predictions
        for prediction in beam_predictions: # Iterate through the collection of predictions and print 
                print(f"Measure: {prediction[0]}, Offset: {prediction[1]}, Pitch: {prediction[2]}, Duration: {prediction[3]} needs a beam.")
        else:  # If there are no predicted beams
            print("No Beams Have Been Predicted!")
        return beam_predictions 
model.train()  # Turns training back on after evalutating

def note_comparator(file_path):

    input_test_data = MusicXMLDataset(file_path)
    input_test_loader = DataLoader(input_test_data, batch_size, shuffle=False)

    test_predictions = evaluate_beam_predictions(input_test_loader)
    input_file = parse_musicxml(file_path)

    corrected_list = []

    # Creates a set of notes for comparison from the original file list and the beam_predictions list.
    all_measures = set([test_note[0] for test_note in test_predictions] + [input_note[0] for input_note in input_file])

    # Create two lists of notes from current for-each measure that are either 8ths or 16ths.
    for measure in all_measures:
        test_measure_notes = [test_note for test_note in test_predictions if test_note[0] == measure and test_note[3] in [0.5, 0.25]]
        input_measure_notes = [input_note for input_note in input_file if input_note[0] == measure and input_note[3] in [0.5, 0.25]]

        # Compare the two lists: add the test data if equivalent; the original if not.
        if len(test_measure_notes) == len(input_measure_notes):
            corrected_list.extend(test_measure_notes)
        else:
            corrected_list.extend(input_measure_notes)

    return corrected_list