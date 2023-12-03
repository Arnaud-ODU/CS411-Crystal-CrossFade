
import os
import sys

p = os.path.abspath('../../../../../../../main/Python/edu/odu/cs/cs411/Backend/')
if p not in sys.path:
    sys.path.append(p)

from Song import *
from music21 import *
import unittest

#'Prototype\src\main\Python\edu\odu\cs\cs411\Backend\Data\score.xml'

    
path = os.path.abspath('../../../../../../../main/Python/edu/odu/cs/cs411/Backend/Data/score.xml')

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
    
    def test_add_beams(self):
        score = Song(path)
        score.parsed_music.parts[0].measure(3).notesAndRests[0].beams.beamsList = []
        self.assertEqual(len(score.parsed_music.parts[0].measure(3).notesAndRests[0].beams), 0)
        score.add_beams(1, 3, 1, 1, 0, 0)
        self.assertEqual(len(score.parsed_music.parts[0].measure(3).notesAndRests[0].beams), 1)
        self.assertEqual(score.parsed_music.parts[0].measure(3).notesAndRests[0].beams.getTypeByNumber(1), 'start')
        
    def test_mark_error(self):
        score = Song(path)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].step, 'B')
        score.mark_error(1,2,2,1)
        self.assertEqual(score.parsed_music.parts[0].measure(2).notesAndRests[1].lyric, "Error #1")
        
    def test_change_duration(self):
        score = Song(path)
        self.assertEqual(score.parsed_music.parts[0].measure(1).notesAndRests[0].duration.type, 'quarter')
        score.change_duration(1, 1, 1, 'half', 0)
        self.assertEqual(score.parsed_music.parts[0].measure(1).notesAndRests[0].duration.type, 'half')
        

def test_song_changes():
    unittest.main()

unittest.main()
        