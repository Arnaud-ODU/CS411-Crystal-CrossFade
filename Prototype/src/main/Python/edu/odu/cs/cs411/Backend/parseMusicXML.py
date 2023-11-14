import music21
import webcolors
from music21 import  * 
# Mohamed Abdullahi
# CS411
 
 # Extract music information and present it as a string
def parsemusic_XML(file):
    score = converter.parse(file)
    complete_stringfinal = [] # Create a list with all notes with their info
    # We want a string of pitch, duration type, duration-quarter length, tie ,pitch space and octave
    def concat_musicinfo(note_info):
        notation_string = ''; # Collect all note info into one string
        
        tie_string=''; # Will return start, continue or stop if there are tied notes
        if x.tie != None: # Check if notes are tied or not
           tie_string = x.tie.type;
       
        #pitch.ps can distinguish between sharps and flats making it better than pitch.midi
        notation_string = str(note_info.pitch) + ", " + str(note_info.duration.type) + ", " + str(note_info.duration.quarterLength)+", " +str(tie_string) + ", " + str(note_info.pitch.ps) + ", " + str(note_info.octave);
         
        return notation_string
    
    for note in score.recurse().notes:
       #Check if note is a note or chord(A collection of 3 notes or more being played at the same time)
        if (note.isNote):
             x = note;
             complete_stringfinal.append(concat_musicinfo(x))
          
    
        if (note.isChord):
            for x in note._notes:
             complete_stringfinal.append(concat_musicinfo(x))             
          
        
    return (complete_stringfinal)