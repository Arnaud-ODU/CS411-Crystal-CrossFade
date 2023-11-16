#This File Will Be Used For Unit Tests For The Song Class
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy

import os
import sys

p = os.path.abspath('../../../../../../../main/Python/edu/odu/cs/cs411/Backend')
if p not in sys.path:
    sys.path.append(p)

from Song import *
import unittest

class Test_SongMethods(unittest.TestCase):

    #Tests Default Constructor (Sanity Check)
    def test_DefaultConstructor(self):  
        song = Song()
        self.assertEqual(song.tracks, [])
        self.assertEqual(song.keys, [])
        self.assertEqual(song.tempos, [])
        self.assertEqual(song.time_signatures, [])
    

    #Tests The SetVars Function
    def test_SetVars(self):
        song = Song()
        track = Track()
        note1 = Note()
        note2 = Note()
        note1.SetVars('A4', 440.0, 1, 1)
        note2.SetVars('B4', 493.883, 1, 1)
        track.SetVars('Violin', [note1, note2], ['Treble'])
        song.SetVars([track], ['C major'], [60], ['4/4'])
        
        self.assertEqual(song.tracks[0], track)
        self.assertEqual(note1, song.tracks[0].notes[0])
        self.assertEqual(note2, song.tracks[0].notes[1])
        self.assertEqual(song.tracks[0].clefs[0], 'Treble')
        self.assertEqual(song.tracks[0].instrument, 'Violin')
        
        self.assertEqual(song.keys[0], 'C major')
        self.assertEqual(song.tempos[0], 60)
        self.assertEqual(song.time_signatures[0], '4/4')
        

#Runs All Tests Present In The File
def test_Song():
    unittest.main()
