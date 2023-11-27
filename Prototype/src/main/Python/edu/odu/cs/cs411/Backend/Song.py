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
import copy


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
        
    def export_musicxml(self, Save_As):
        """Given A Path To Save The File To, Exports Data As A MusicXML File"""
        self.parsed_music.write("musicxml", Save_As)
        
    def export_midi(self, Save_As):
        """Given A Path To Save The File To, Exports Data As A MIDI File"""
        self.parsed_music.write("midi", Save_As)
        
    def change_note_pitch(self, part, measure, note, new_value):
        """Change a note in the parsed music

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
            new_value (_int_, _string_): The new value that the note needs to have
        """
        p = pitch.Pitch(new_value)
        offset = self.parsed_music.parts[part-1].measure(measure).notes[note-1].offset
        n = copy.deepcopy(self.parsed_music.parts[part-1].measure(measure).notesAndRests[note-1])
        n.pitch = p
        self.parsed_music.parts[part-1].measure(measure).remove(self.parsed_music.parts[part-1].measure(measure).notes[note-1])
        self.parsed_music.parts[part-1].measure(measure).insert(offset, n)
        
    def increase_pitch_by_semitone(self, part, measure, note):
        """Increases the pitch of a note by a semitone given the part, measure and position inside the measure of a note.

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
        """
        self.parsed_music.parts[part-1].measure(measure).notes[note-1].transpose(1, inPlace=True)
        
    def decrease_pitch_by_semitone(self, part, measure, note):
        """Decreases the pitch of a note by a semitone given the part, measure and position inside the measure of a note.

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
        """
        self.parsed_music.parts[part-1].measure(measure).notes[note-1].transpose(-1, inPlace=True)
          