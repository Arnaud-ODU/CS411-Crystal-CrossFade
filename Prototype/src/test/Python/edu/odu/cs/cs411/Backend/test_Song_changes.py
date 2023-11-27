from Prototype.src.main.Python.edu.odu.cs.cs411.Backend.Song import *
from music21 import *
import unittest
import os

#'Prototype\src\main\Python\edu\odu\cs\cs411\Backend\Data\score.xml'
path = os.path.join("Prototype", "src", "main", "Python", "edu", "odu", "cs", "cs411", "Backend", "Data", "score.xml")

class Test_Song_Modifiers(unittest.TestCase):
    
    # Checks that the note pitch has been changed to the requested value
    def test_change_note_pitch(self):
        score = Song(path)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'B')
        score.change_note_pitch(1, 2, 2, 'C4')
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'C')
    
    # Checks that the note pitch has been increased by a semitone   
    def test_note_semitone_increase(self):
        score = Song(path)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'B')
        score.increase_pitch_by_semitone(1, 2, 2)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].nameWithOctave, 'C5')
    
    # Checks that the note pitch has been increased by a semitone   
    def test_note_semitone_decrease(self):
        score = Song(path)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'B')
        score.decrease_pitch_by_semitone(1, 2, 2)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].nameWithOctave, 'B-4')
        
unittest.main()
        