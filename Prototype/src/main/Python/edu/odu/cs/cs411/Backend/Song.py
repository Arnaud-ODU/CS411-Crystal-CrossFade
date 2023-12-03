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
#@author Joseph Wassell, Virginia Vano Rano
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

    #WIP
    #def add_starting_beam(self, part, measure, note):
        """Adds the starting beams required for the provided note

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
        """
        #noteType = self.parsed_music.parts[part-1].measure(measure).notes[note-1].duration.type
        #self.parsed_music.parts[part-1].measure(measure).notes[note-1].beams.fill(noteType, type = 'start')

    #WIP
    #def add_middle_beam(self, part, measure, note, prevType, nextType):
        """Adds the middle beams required for the provided note

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
            prevType (_str_): The type of the previous note (eighth, 16th, 32nd...)
            nextType (_str_): The type of the next note (eighth, 16th, 32nd...)
        """
        
        #n = self.parsed_music.parts[part-1].measure(measure).notes[note-1]
        
        # Removes beams in case there are any already placed
        #n.beams.beamList = []
        
        #noteType = n.duration.type
        
        #if prevType == noteType and nextType == noteType:            
        #    n.beams.fill(noteType, type = 'continue')
            
        #noteType = self.number_of_beams(noteType)
        #prevType = self.number_of_beams(prevType)
        #nextType = self.number_of_beams(nextType)
        
        #if prevType < noteType:
        #    for x in range(prevType):
        #        n.beams.append('continue')
        #    for x in range(noteType-prevType):
        #        n.beams.append('start')
                
        #if prevType > noteType:
        #    for x in range(prevType):
        #        n.beams.append('continue')
        #    for x in range(noteType-prevType):
        #        n.beams.append('stop')

    #WIP
    #def add_end_beam(self, part, measure, note):
        """Adds the final beams required for the provided note

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
        """
        #noteType = self.parsed_music.parts[part-1].measure(measure).notes[note-1].duration.type
        #self.parsed_music.parts[part-1].measure(measure).notes[note-1].beams.fill(noteType, type = 'stop')

    def add_beams(self, part, measure, note, start, middle, end):
        """Adds the specified number of beams of each type to a note. This function does not work for partial beams

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
            start (_int_): The number of beams of type 'start' that have to be added
            middle (_int_): The number of beams of type 'continue' that have to be added
            end (_int_): The number of beams of type 'stop' that have to be added
        """
        n =  self.parsed_music.parts[part-1].measure(measure).notes[note-1]
        
        if middle > 0:
            for i in range(middle):
                n.beams.append('continue')
            
        if start > 0:
            for i in range(start):
                n.beams.append('start')
                
        if end > 0:
            for i in range(end):
                n.beams.append('stop')
                
    #def add_beams_between_notes(self, part, measure, notes):
        """Given a list of notes, it inserts simple beams between notes. It does not include partial beams.

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            notes (_list of int_): A list with the numbers of the notes. To connect the first 4 notes in a measure, this list would contain the numbers 1, 2, 3 and 4
        """
        
        #self.add_starting_beam(part, measure, notes[0])
        
    #def number_of_beams(self, noteType):
        #if noteType == 'eighth' or noteType == '8th':
        #    return 2
        
        #if noteType == '16th':
        #    return 3
        
        #if noteType == '32nd':
        #    return 4

        
    def mark_error(self, part, measure, note, error_number):
        """Given A Specifc Note And An Error Number Marks The Note As An Error.
        
        Args:
            part (_int_): The Number Of The Part Where The Note Is Located
            measure (_int_): The Number Of The Measure Which Containes The Note
            note (_int_): The Number Note That Should Be Marked As An Error
            error_number (_int_): The Number Error That Was Found In The Song
        """
        n =  self.parsed_music.parts[part-1].measure(measure).notes[note-1]
        n.lyric ="Error #" + str(error_number)
        
    def change_duration(self, part, measure, note, neededDuration, neededDots):
        """Changes the duration of the specified note.

        Args:
            part (_int_): The number of the part where the note is located, where the first part would be 1
            measure (_int_): The number of the measure where the note is located
            note (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
            d (_str_): The type of note that represents the duration that the note needs to be. If the note has to be a half note, this would be 'half'
            dots (_int_): The number of dots the note has
        """
        n = self.parsed_music.parts[part-1].measure(measure).notesAndRests[note-1]
        n.duration = duration.Duration(type=neededDuration, dots=neededDots)
