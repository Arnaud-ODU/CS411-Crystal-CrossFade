import mido
from mido import MidiFile, MidiTrack, Message

# Create a new MIDI file
mid = MidiFile()

# Add a track
track = MidiTrack()
mid.tracks.append(track)

# Add some notes to the track
notes = [60, 62, 64, 65, 67, 69, 71]  # C, D, E, F, G, A, B

for note in notes:
    # Note on event
    track.append(Message('note_on', note=note, velocity=64, time=480))
    # Note off event
    track.append(Message('note_off', note=note, velocity=64, time=480))

# Save the MIDI file
output_path = "../../../../../../output/output.mid"
mid.save(output_path)

print("MIDI file exported as 'output.mid'")
