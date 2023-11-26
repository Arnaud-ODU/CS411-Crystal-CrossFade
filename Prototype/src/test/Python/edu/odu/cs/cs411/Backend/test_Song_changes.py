from Prototype.src.main.Python.edu.odu.cs.cs411.Backend.Song import *
from music21 import *
import unittest

class Test_Song_Modifiers(unittest.TestCase):
    
    # Checks that the note has been changed appropriately
    def test_change_note_pitch(self):
        score = Song('Prototype\src\main\Python\edu\odu\cs\cs411\Backend\Data\score.xml')
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'B')
        score.change_note_pitch(1, 2, 2, 'C4')
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'C')
        
unittest.main()
        