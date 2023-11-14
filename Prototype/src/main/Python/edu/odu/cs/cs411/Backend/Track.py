#Track Class
#This Class Will Be Used To Store And Access Tracks As Well As
#Information Contained Within Those Track
#
#In Music A Track Is Typically The Part Of A Single Instrument
#
#A Track Has:
#  An Instrument (The Instrument That Plays The Stored Notes)
#  Notes (Multiple Notes To Be Played By The Instrument)
#  Clefs (The Musical Clef(s) That The Instrument Uses)
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy


import Note


class Track(object):
    
    #Default Constructor
    def __init__(self):
        self.instrument = ""
        self.notes = []
        self.clefs = []
        
    #Non-Default Constructor    
    def __init__(self, instrument, notes, clefs):
        self.instrument = instrument
        self.notes = notes
        self.clefs = clefs
        
    #Copy Constructor
    def __init__(self, rhs):
        self.instrument = rhs.instrument
        self.notes = rhs.notes
        self.clefs = rhs.clefs
        
    #Print Value
    def __str__(self):
        temp = "Track info:\n\tInstrument: {}"
        for i in range(0, len(clefs)):
            temp += "\n\tClef " + str(i) + ": {}".format(str(self.clefs[i])) 
        return temp.format(self.instrument, self.clefs)
       
    #__eq__ Operator (==)
    def __eq__(self, rhs):
        if self.instrument == rhs.instrument:
            if len(self.notes) == len(rhs.notes):
                is_equal = True
                for i in range(0, len(self.notes)):
                    if not self.notes[i] == rhs.notes[i]:
                        is_equal = False
                if is_equal:
                    if len(self.clefs) == len(rhs.clefs):
                        for i in range(0, len(self.clefs)):
                            if not self.clefs[i] in rhs.clefs:
                                is_equal = False
            return True
        return False
    
               
    
        
        
    