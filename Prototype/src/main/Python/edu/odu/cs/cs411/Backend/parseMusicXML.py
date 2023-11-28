import music21
import webcolors
from music21 import  * 
# Mohamed Abdullahi
# CS411
 
def parsemusic_XML(file):
    """Extract music information from a MusicXML file.
    
    Args:
        file: The MusicXML file being extracted from.
        note_info: Information including pitch, duration type,
        duration-quarter length, tie, pitch space, and octave, from an
        individual note.
    
    Returns:
        notation_string: A concatenated string containing all of the information 
        from note_info. pitch.ps is used to distinguish between sharps and flats.
        complete_stringfinal: A list of every notation_string from concat_musicinfo.
    """
    score = converter.parse(file)
    complete_stringfinal = [] 

    def concat_musicinfo(note_info):
        notation_string = ''; 
        
        tie_string=''; 
        if x.tie != None: 
            """Check if notes are tied or not"""
            tie_string = x.tie.type;
       
        notation_string = str(note_info.pitch) + ", " + str(note_info.duration.type) + ", " + str(note_info.duration.quarterLength)+", " +str(tie_string) + ", " + str(note_info.pitch.ps) + ", " + str(note_info.octave);
         
        return notation_string
    
    for note in score.recurse().notes:
        """Check if note is a note or chord (a collection of three notes or more being played at the same time)"""
        if (note.isNote):
             x = note;
             complete_stringfinal.append(concat_musicinfo(x))
          
    
        if (note.isChord):
            for x in note._notes:
             complete_stringfinal.append(concat_musicinfo(x))             
          
        
    return (complete_stringfinal)