#Song Class
#This Class Will Be Used To Store And Access Tracks As Well As
#Information Contained Within Those Track
#
#In Music A Track Is Typically The Part Of A Single Instrument
#
#A Song Has:
#  A Track(s) (The Part Of A Single Instrument)
#  A Key(s) (The Rule For The Notation Used For Displaying Notes)
#  A Tempo(s) (The Speed At Which The Notes Are Played)
#  A Time Signature(s) (The Ruling For How The Notation Interacts With The Tempo)
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy


from Track import *


class Song(object):
    
    #Default Constructor
    def __init__(self):
        self.tracks = []
        self.keys = []
        self.tempos = []
        self.time_signatures = []
        
    #Sets The Vars For The Song
    def SetVars(self, tracks, keys, tempos, time_signatures):
        self.tracks = tracks
        self.keys = keys
        self.tempos = tempos
        self.time_signatures = time_signatures