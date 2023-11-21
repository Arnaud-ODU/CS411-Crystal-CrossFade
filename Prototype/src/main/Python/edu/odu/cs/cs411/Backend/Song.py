#Song Class
#This Class Will Be Used To Store And Access Tracks As Well As
#Information Contained Within Those Track
#
#In Music A Track Is Typically The Part Of A Single Instrument
#
#A Song Has:
#  A MusicXML file path (path)
#  A music21.stream (parsed_work) score For Storing Parsed Data
#
#@author Joseph Wassell
#CS 411W Prof. Kennedy


from music21 import *


class Song(object):
    
    
    def __init__(self, path='Invalid_Path'):
        """Default Constructor"""
        if path not in 'Invalid_Path':
            self.path = path
            self.parsed_music = converter.parse(path)
        else:
            self.parsed_music = None
            
            
    def import_musicxml(self, file_path):
        """Given A MusicXML File Path Initializes Song Data"""
        self.path = file_path
        self.parsed_music = converter.parse(file_path)
        
    def export_musicxml(self, Save_As)
        """Given A Path To Save The File To, Exports Data As A MusicXML"""
        