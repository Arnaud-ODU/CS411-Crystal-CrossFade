#This File Will Be Used For Unit Tests For The Note Class
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy

import os
import sys

p = os.path.abspath('../../../../../../../main/Python/edu/odu/cs/cs411/Backend')
if p not in sys.path:
    sys.path.append(p)

from Note import *
import unittest

class Test_NoteMethods(unittest.TestCase):

    #Tests Default Constructor (Sanity Check)
    def test_DefaultConstructor(self):  
        note = Note()
        self.assertEqual(note.name, '')
        self.assertEqual(note.pitch, -1)
        self.assertEqual(note.duration, -1)
        self.assertEqual(note.amplitude, -1)
    

    #Tests The Nondefault Constructor
    #def test_NondefaultConstructor(self):
    #    note = Note('john', 1, 2, 3)
    #    self.assertEqual(note.name, 'john')
    #    self.assertEqual(note.pitch, 1)
    #    self.assertEqual(note.duration, 2)
    #    self.assertEqual(note.amplitude, 3)
    
    #Tests The Copy Constructor
    #def test_CopyConstructor(self):
    #    note1 = Note('smith', 4, 5, 6)
    #    note2 = Note(note1)
    #    self.assertEqual(note1.name, note2.name)
    #    self.assertEqual(note1.pitch, note2.pitch)
    #    self.assertEqual(note1.duration, note2.duration)
    #    self.assertEqual(note1.amplitude, note2.amplitude)
    
    #Tests The SetVars Function
    def test_SetVars(self):
        note = Note()
        note.SetVars('John', 1, 2, 3)
        self.assertEqual(note.name, 'John')
        self.assertEqual(note.pitch, 1)
        self.assertEqual(note.duration, 2)
        self.assertEqual(note.amplitude, 3)
        

#Runs All Tests Present In The File
def test_Note():
    unittest.main()
    #test_DefaultConstructor()
    #test_NondefaultConstructor()
    #test_CopyConstructor()
    
    
