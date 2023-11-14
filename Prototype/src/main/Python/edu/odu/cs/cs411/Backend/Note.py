#Note Class
#This Class Will Be Used To Store And Access Notes As Well As
#Information Contained Within Those Notes
#A Note Has:
#  A Name (What The Note Is Refered To As Ex: b flat)
#  A Pitch (What The Note Sounds Like)
#  A Duration (How Long The Note Lasts)
#  An Amplitude (How Loud The Note Is)
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy

class Note(object):
    
    #Default Constructor
    def __init__(self):
        self.name = ""
        self.pitch = -1
        self.duration = -1
        self.amplitude = -1
        
    #Non-Default Constructor    
    def __init__(self, name, pitch, duration, amplitude):
        self.name = name
        self.pitch = pitch
        self.duration = duration
        self.amplitude = amplitude
        
    #Copy Constructor
    def __init__(self, rhs):
        self.name = rhs.name
        self.pitch = rhs.pitch 
        self.duration = rhs.duration
        self.amplitude = rhs.amplitude
        
    #Print Value
    def __str__(self):
        temp = "Note info:\n\tName: {}\n\tPitch: {}\n\tDuration: {}\n\tAmplitude: {}"
        return temp.format(self.name, self.pitch, self.duration, self.amplitude)
       
    #__eq__ Operator (==)
    def __eq__(self, rhs):
        if (self.name == rhs.name and
            self.pitch == rhs.pitch and
            self.duration == rhs.duration and
            self.amplitude == rhs.amplitude):
            return True
        return False
    
               
    
        
        
    