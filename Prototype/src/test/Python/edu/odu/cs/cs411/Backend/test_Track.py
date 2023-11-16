#This File Will Be Used For Unit Tests For The Track Class
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy

import os
import sys

p = os.path.abspath('../../../../../../../main/Python/edu/odu/cs/cs411/Backend')
if p not in sys.path:
    sys.path.append(p)

from Track import *
import unittest

class Test_TrackMethods(unittest.TestCase):

    #Tests Default Constructor (Sanity Check)
    def test_DefaultConstructor(self):  
        track = Track()
        self.assertEqual(track.instrument, '')
        self.assertEqual(track.notes, [])
        self.assertEqual(track.clefs, [])
    

    #Tests The SetVars Function
    def test_SetVars(self):
        track = Track()
        note1 = Note()
        note2 = Note()
        note1.SetVars('A4', 440.0, 1, 1)
        note2.SetVars('B4', 493.883, 1, 1)
        track.SetVars('Violin', [note1, note2], ['Treble'])
        self.assertEqual(track.instrument, 'Violin')
        self.assertEqual(note1, track.notes[0])
        self.assertEqual(note2, track.notes[1])
        self.assertEqual(track.clefs[0], 'Treble')
        

#Runs All Tests Present In The File
def test_Track():
    unittest.main()
    #test_DefaultConstructor()
    #test_NondefaultConstructor()
    #test_CopyConstructor()
    
    
