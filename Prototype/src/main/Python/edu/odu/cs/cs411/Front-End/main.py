# Import necessary modules
from tkinter import *
from customtkinter import *
from PIL import Image, ImageTk
from configparser import ConfigParser
import os
import music21 as m21

us = m21.environment.UserSettings()
us_path = us.getSettingsPath()
if not os.path.exists(us_path):
    us.create()
us['musescoreDirectPNGPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'
us['musicxmlPath'] = r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'

import sys

# Add paths to the system for custom modules
sys.path.append('Prototype/src/main/Python/edu/odu/cs/cs411')
sys.path.append('Prototype/src/main/Python/edu/odu/cs/cs411/Backend')

# Import a module from the custom paths
from Backend.parseMusicXML import parsemusic_XML

# Define the main application class
class App(CTk):

    # Constructor for the application
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize various variables and settings
        self.var_view_signatures = IntVar(value=1)
        self.var_view_dynamics = IntVar(value=1)
        self.var_view_duration = IntVar(value=1)
        self.var_view_key = IntVar(value=1)
        self.var_view_transpose = IntVar(value=1)
        self.configurator = ConfigParser()
        self.title('CrossFade Main Menu')
        self.help_wraplength = 300  # Width of labels in the help side panel

        # Set up menu, time signatures, and keys
        self.add_menu()
        self.time_signatures = (
            '2/4',
            '3/4',
            '4/4'
        )
        self.minor_keys = (
            'A#',
            'D#',
            'G#',
            'C#',
            'F#',
            'B',
            'E',
            'A',
            'D',
            'G',
            'C',
            'F',
            'Bb',
            'Eb',
            'Ab'
        )
        self.major_keys = (
            'C#',
            'F#',
            'B',
            'E',
            'A',
            'D',
            'G',
            'C',
            'F',
            'Bb',
            'Eb',
            'Ab',
            'Db',
            'Gb',
            'Cb'
        )
        self.d_major = {
            'C#': ('C', 'D'),
            'F#': ('F', 'G'),
            'B': ('Bb', 'C'),
            'E': ('Eb', 'F'),
            'A': ('Ab', 'Bb'),
            'D': ('Db', 'Eb'),
            'G': ('Gb', 'Ab'),
            'C': ('B', 'Db'),
            'F': ('E', 'F#'),
            'Bb': ('A', 'B'),
            'Eb': ('D', 'E'),
            'Ab': ('G', 'A'),
            'Db': ('C', 'D'),
            'Gb': ('F#', 'G'),
            'Cb': ('B', 'C')
        }
        self.d_minor = {
            'A#': ('A', 'B'),
            'D#': ('D', 'E'),
            'G#': ('G', 'A'),
            'C#': ('C', 'D'),
            'F#': ('F', 'G'),
            'B': ('Bb', 'C'),
            'E': ('Eb', 'F'),
            'A': ('G#', 'Bb'),
            'D': ('C#', 'D#'),
            'G': ('F#', 'G#'),
            'C': ('B', 'C#'),
            'F': ('E', 'F#'),
            'Bb': ('A', 'B'),
            'Eb': ('D', 'E'),
            'Ab': ('G', 'A')
        }
        self.var_time_signatures = StringVar()
        self.var_keys = StringVar()
        self.var_mode_keys = StringVar(value='minor')
        self.var_transpose = StringVar()
        self.var_transpose_mode = StringVar(value='minor')

        # Set up images for buttons
        self.img_wholenote = CTkImage(
            light_image=Image.open("Images/wholenote.png"),
            size=(30, 30)
        )
        self.img_halfnote = CTkImage(
            light_image=Image.open("Images/halfnote.png"),
            size=(30, 30)
        )
        self.img_quarternote = CTkImage(
            light_image=Image.open("Images/quarternote.png"),
            size=(30, 30)
        )
        self.img_eighthnote = CTkImage(
            light_image=Image.open("Images/eighthnote.png"),
            size=(30, 30)
        )
        self.img_sixteenthnote = CTkImage(
            light_image=Image.open("Images/sixteenthnote.png"),
            size=(30, 30)
        )
        self.img_thirtysecondnote = CTkImage(
            light_image=Image.open("Images/thirtysecondnote.png"),
            size=(30, 30)
        )
        self.img_play = CTkImage(
            light_image=Image.open("Images/play.png"),
            size=(30, 30)
        )
        self.img_pause = CTkImage(
            light_image=Image.open("Images/pause.png"),
            size=(30, 30)
        )
        self.img_stop = CTkImage(
            light_image=Image.open("Images/stop.png"),
            size=(30, 30)
        )
        self.img_redo = CTkImage(
            light_image=Image.open("Images/redo.png"),
            size=(30, 30)
        )
        self.img_undo = CTkImage(
            light_image=Image.open("Images/undo.png"),
            size=(30, 30)
        )
        self.img_comment = CTkImage(
            light_image=Image.open("Images/comment.png"),
            size=(30, 30)
        )

        # Add settings, read configuration, and set up toolbar and timeline
        self.add_settings()
        self.read_config()
        self.add_toolbar()
        self.add_timeline()
        self.add_help()
        self.after(1, self.maximize)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

    
    # Method to create the help frame without inserting it
    def add_help(self):
        self.frame_help = CTkFrame(self)


    # Method to add the toolbar
    def add_toolbar(self):
        self.frame_toolbar = CTkFrame(self)
        self.frame_toolbar.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        self.frame_toolbar.grid(
            row=0,
            column=1,
            pady=(10,5),
            padx=(0,10),
            sticky='NSEW',
        )
        self.button_play = CTkButton(
            self.frame_toolbar, 
            image=self.img_play,
            width=0,
            text=''
        )
        self.button_play.grid_configure(
            row=0,
            column=0,
            padx=(10,3),
            pady=10,
            sticky='EW'
        )
        self.button_pause = CTkButton(
            self.frame_toolbar, 
            image=self.img_pause,
            width=0,
            text=''
        )
        self.button_pause.grid_configure(
            row=0,
            column=1,
            padx=3,
            pady=10,
            sticky='EW'
        )
        self.button_stop = CTkButton(
            self.frame_toolbar, 
            image=self.img_stop,
            width=0,
            text=''
        )
        self.button_stop.grid_configure(
            row=0,
            column=2,
            padx=3,
            pady=10,
            sticky='EW'
        )
        self.button_redo = CTkButton(
            self.frame_toolbar, 
            image=self.img_redo,
            width=0,
            text=''
        )
        self.button_redo.grid_configure(
            row=0,
            column=3,
            padx=3,
            pady=10,
            sticky='EW'
        )
        self.button_undo = CTkButton(
            self.frame_toolbar, 
            image=self.img_undo,
            width=0,
            text=''
        )
        self.button_undo.grid_configure(
            row=0,
            column=4,
            padx=3,
            pady=10,
            sticky='EW'
        )
        self.button_comment = CTkButton(
            self.frame_toolbar, 
            image=self.img_comment,
            width=0,
            text=''
        )
        self.button_comment.grid_configure(
            row=0,
            column=5,
            padx=(3,10),
            pady=10,
            sticky='EW'
        )

    # Method to add the timeline
    def add_timeline(self):
        self.frame_timeline = CTkFrame(self)
        self.frame_timeline.grid(
            row=1,
            column=1,
            pady=(5,10),
            padx=(0,10),
            sticky='NSEW',
        )
        self.canvas_timeline = Canvas(
            self.frame_timeline,
            scrollregion=(0,0,500,500)
        )
        self.canvas_scrollbar=CTkScrollbar(
            self.frame_timeline,
            orientation='vertical', 
            command=self.canvas_timeline.yview
        )
        self.canvas_scrollbar.pack(side=RIGHT,fill=Y)
        self.canvas_timeline.config(yscrollcommand=self.canvas_scrollbar.set)
        self.canvas_timeline.pack(
            side=LEFT,expand=True,fill=BOTH
        )
        self.canvas_frame=Canvas(self.canvas_timeline)
        self.canvas_frame.bind("<Configure>",lambda e: self.canvas_timeline.configure(scrollregion=self.canvas_timeline.bbox("all")))
        self.canvas_timeline.create_window((0, 0), window=self.canvas_frame, anchor="nw")

    # Method to read configuration from a file
    def read_config(self):
        if os.path.exists('config.ini'):
            self.configurator.read('config.ini')
            set_appearance_mode(self.configurator['view']['theme'])
            self.var_view_signatures.set(self.configurator['view']['signatures'])
            if not self.var_view_signatures.get():
                self.frame_signatures.grid_forget()
            self.var_view_dynamics.set(self.configurator['view']['dynamics'])
            if not self.var_view_dynamics.get():
                self.frame_dynamics.grid_forget()
            self.var_view_duration.set(self.configurator['view']['duration'])
            if not self.var_view_duration.get():
                self.frame_notes.grid_forget()
            self.var_view_key.set(self.configurator['view']['key'])
            if not self.var_view_key.get():
                self.frame_key.grid_forget()
            self.var_view_transpose.set(self.configurator['view']['transpose'])
            if not self.var_view_transpose.get():
                self.frame_transpose.grid_forget()
        else:
            self.configurator['view'] = {
                'theme': 'dark',
                'signatures': 1,
                'dynamics': 1,
                'duration': 1
            }
 
    # Method to add various settings
    def add_settings(self):
        self.frame_settings = CTkFrame(self)
        self.frame_settings.grid(
            row=0,
            rowspan=2,
            column=0,
            sticky='NSEW',
            padx=10,
            pady=10
        )
        self.frame_settings.grid_columnconfigure(0, weight=1)
        self.frame_signatures = CTkFrame(
            self.frame_settings,
        )
        self.frame_signatures.grid(
            row=0,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        self.frame_signatures.grid_columnconfigure(0, weight=1)
        CTkLabel(
            self.frame_signatures, 
            text='Time Signatures', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            sticky='NEW',
            padx=5,
            pady=5
        )
        CTkOptionMenu(
            self.frame_signatures,
            values=self.time_signatures,
            command=self.time_signature_clicked,
            variable=self.var_time_signatures
        ).grid(
            row=1,
            column=0,
            padx=5,
            pady=10
        )
        self.frame_dynamics = CTkFrame(
            self.frame_settings,
        )
        self.frame_dynamics.grid(
            row=2,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        self.frame_dynamics.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        CTkLabel(
            self.frame_dynamics, 
            text='Dynamics', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            columnspan=6,
            sticky='NEW',
            padx=5,
            pady=5
        )
        self.button_pianissimo = CTkButton(
            self.frame_dynamics,
            text='pp',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('pianissimo')
        )
        self.button_pianissimo.grid(
            row=1,
            column=0,
            pady=10
        )
        self.button_piano = CTkButton(
            self.frame_dynamics,
            text='p',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('piano')
        )
        self.button_piano.grid(
            row=1,
            column=1,
            pady=10
        )
        self.button_mezopiano = CTkButton(
            self.frame_dynamics,
            text='mp',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('mezopiano')
        )
        self.button_mezopiano.grid(
            row=1,
            column=2,
            pady=10
        )
        self.button_mezoforte = CTkButton(
            self.frame_dynamics,
            text='mf',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('mezoforte')
        )
        self.button_mezoforte.grid(
            row=1,
            column=3,
            pady=10
        )
        self.button_forte = CTkButton(
            self.frame_dynamics,
            text='f',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('forte')
        )
        self.button_forte.grid(
            row=1,
            column=4,
            pady=10
        )
        self.button_fortissimo = CTkButton(
            self.frame_dynamics,
            text='ff',
            font=('Kristen ITC', 24, 'bold'),
            text_color='black',
            width=45,
            command=lambda:self.dynamics_clicked('fortissimo')
        )
        self.button_fortissimo.grid(
            row=1,
            column=5,
            pady=10
        )
        CTkSlider(
            self.frame_dynamics, 
            from_=0,
            to=100, 
            command=self.slider_dynamics_moved
        ).grid(
            row=2,
            column=0,
            columnspan=6,
            padx=10,
            pady=10
        )
        self.frame_notes = CTkFrame(
            self.frame_settings,
        )
        self.frame_notes.grid(
            row=3,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        self.frame_notes.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        CTkLabel(
            self.frame_notes, 
            text='Notes Duration', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            columnspan=6,
            sticky='NEW',
            padx=5,
            pady=5
        )
        self.button_wholenote = CTkButton(
            self.frame_notes, 
            image=self.img_wholenote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('wholenote')
        )
        self.button_wholenote.grid(
            row=1,
            column=0,
            pady=10
        )
        self.button_halfnote = CTkButton(
            self.frame_notes, 
            image=self.img_halfnote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('halfnote')
        )
        self.button_halfnote.grid(
            row=1,
            column=1,
            pady=10
        )
        self.button_quarternote = CTkButton(
            self.frame_notes, 
            image=self.img_quarternote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('quarternote')
        )
        self.button_quarternote.grid(
            row=1,
            column=2,
            pady=10
        )
        self.button_eighthnote = CTkButton(
            self.frame_notes, 
            image=self.img_eighthnote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('eighthnote')
        )
        self.button_eighthnote.grid(
            row=1,
            column=3,
            pady=10
        )
        self.button_sixteenthnote = CTkButton(
            self.frame_notes, 
            image=self.img_sixteenthnote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('sixteenthnote')
        )
        self.button_sixteenthnote.grid(
            row=1,
            column=4,
            pady=10
        )
        self.button_thirtysecondnote = CTkButton(
            self.frame_notes, 
            image=self.img_thirtysecondnote,
            text='',
            width=1,
            command=lambda:self.notes_clicked('thirtysecondnote')
        )
        self.button_thirtysecondnote.grid(
            row=1,
            column=5,
            pady=10
        )
        self.frame_key = CTkFrame(
            self.frame_settings,
        )
        self.frame_key.grid(
            row=4,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        self.frame_key.grid_columnconfigure((0,1), weight=1)
        CTkLabel(
            self.frame_key, 
            text='Change key', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky='NEW',
            padx=5,
            pady=(5,0)
        )
        CTkLabel(
            self.frame_key, 
            text='(without moving the notes)', 
            font=('Helvetica', 12),
            text_color='grey'
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky='NEW',
            padx=5,
            pady=(0,5)
        )
        self.dropdown_keys = CTkOptionMenu(
            self.frame_key,
            values=self.minor_keys,
            command=self.keys_clicked,
            variable=self.var_keys
        )
        self.dropdown_keys.grid(
            row=2,
            column=0,
            padx=5,
            pady=10
        )
        self.dropdown_keys_mode = CTkOptionMenu(
            self.frame_key,
            values=('minor', 'major'),
            command=self.keys_mode_clicked,
            variable=self.var_mode_keys
        )
        self.dropdown_keys_mode.grid(
            row=2,
            column=1,
            padx=5,
            pady=10
        )
        self.frame_transpose = CTkFrame(
            self.frame_settings,
        )
        self.frame_transpose.grid(
            row=5,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        self.frame_transpose.grid_columnconfigure((0,1,2), weight=1)
        CTkLabel(
            self.frame_transpose, 
            text='Transpose', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky='NEW',
            padx=5,
            pady=(5,0)
        )
        CTkLabel(
            self.frame_transpose, 
            text='(change key and move the notes)', 
            font=('Helvetica', 12),
            text_color='grey'
        ).grid(
            row=1,
            column=0,
            columnspan=3,
            sticky='NEW',
            padx=5,
            pady=(0,5)
        )
        self.dropdown_transpose = CTkOptionMenu(
            self.frame_transpose,
            values=self.minor_keys,
            command=self.transpose_clicked,
            variable=self.var_transpose
        )
        self.dropdown_transpose.grid(
            row=2,
            column=0,
            padx=5,
            pady=10
        )
        self.dropdown_transpose_mode = CTkOptionMenu(
            self.frame_transpose,
            values=('minor', 'major'),
            command=self.transpose_mode_clicked,
            variable=self.var_transpose_mode
        )
        self.dropdown_transpose_mode.grid(
            row=2,
            column=1,
            padx=5,
            pady=10
        )
        self.frame_transpose_buttons = CTkFrame(
            self.frame_transpose,
        )
        self.frame_transpose_buttons.grid(
            row=2,
            column=2,
            padx=5,
            pady=10
        )
        CTkButton(
            self.frame_transpose_buttons,
            text='▲',
            font=('Helvetica', 25),
            width=0,
            command=self.increment_transpose
        ).grid(
            row=0,
            column=0
        )
        CTkButton(
            self.frame_transpose_buttons,
            text='▼',
            font=('Helvetica', 25),
            width=0,
            command=self.decrement_transpose
        ).grid(
            row=1,
            column=0
        )

    def notes_clicked(self, note):
        self.display_help()
        CTkLabel(
            self.frame_help,
            text='Steps to Change Note Duration:',
            font=('Helvetica', 16, 'bold')
        ).grid(
            row=0, 
            column=0,
            padx=10,
            pady=10
        )
        CTkLabel(
            self.frame_help,
            text='1.Selecting a Note or Passage:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=1, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='Click on a single note or drag across a range of notes where you want to apply duration changes',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=2, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='2.Accessing Duration Options:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=3, 
            column=0,
            padx=10,
            pady=(10,0)
        )
        CTkLabel(
            self.frame_help,
            text='Right-click on the selected note(s) or use the toolbar menu to open the note duration option.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=4, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Choosing a New Duration:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=5, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Select the desired note value (e.g., half note, quarter note) from the toolbar The selected note(s) will update to the new duration.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=6, 
            column=0,
            padx=10,
        )

    # Method that clears and displays the help side panel
    def display_help(self):
        for child in self.frame_help.winfo_children():
            child.grid_forget()
        self.frame_help.grid(
            row=0,
            rowspan=2, 
            column=2,
            sticky='NSEW'
        )

    # Method that triggers when dynamics buttons are clicked
    def dynamics_clicked(self, type_):
        self.show_dynamics_help()

    def show_dynamics_help(self):
        self.display_help()
        CTkLabel(
            self.frame_help,
            text='Steps to Change Dynamics:',
            font=('Helvetica', 16, 'bold')
        ).grid(
            row=0, 
            column=0,
            padx=10,
            pady=10
        )
        CTkLabel(
            self.frame_help,
            text='1.Selecting a Note or Passage:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=1, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='Click on a single note or drag across a range of notes where you want to apply dynamic changes.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=2, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='2.Accessing Dynamics Options:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=3, 
            column=0,
            padx=10,
            pady=(10,0)
        )
        CTkLabel(
            self.frame_help,
            text='Right-click on the selected note(s) or use the toolbar menu to open the dynamics options.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=4, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Applying Dynamics:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=5, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Choose from fortissimo (ff) for very loud, forte (f) for loud, mezzo-forte (mf) for moderately loud, mezzo-piano (mp) for moderately quiet, piano (p) for quiet , pianissimo (pp) for very quiet. Click on the desired dynamic symbol to apply it to the selected notes.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=6, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Applying Dynamics:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=7, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Choose from fortissimo (ff) for very loud, forte (f) for loud, mezzo-forte (mf) for moderately loud, mezzo-piano (mp) for moderately quiet, piano (p) for quiet , pianissimo (pp) for very quiet. Click on the desired dynamic symbol to apply it to the selected notes.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=8, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='4.Editing Existing Dynamics:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=9, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Click on an existing dynamic marking to select it. Use the pop-up editor to change the dynamic type or delete it.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=10, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='5.Playback to Review:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=11, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Use the playback feature to listen to how the dynamics affect your music.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=12, 
            column=0,
            padx=10,
        )
  
    # Methods to handle increment and decrement of transpose
    def increment_transpose(self):
        self.show_transpose_help()
        if not self.var_transpose.get():
            return
        if self.var_transpose_mode.get() == 'minor':
            self.var_transpose.set(
                self.d_minor[self.var_transpose.get()][1]
            )
        elif self.var_transpose_mode.get() == 'major':
            self.var_transpose.set(
                self.d_major[self.var_transpose.get()][1]
            )

    def decrement_transpose(self):
        self.show_transpose_help()
        if not self.var_transpose.get():
            return
        if self.var_transpose_mode.get() == 'minor':
            self.var_transpose.set(
                self.d_minor[self.var_transpose.get()][0]
            )
        elif self.var_transpose_mode.get() == 'major':
            self.var_transpose.set(
                self.d_major[self.var_transpose.get()][0]
            )

    # Placeholder method for slider dynamics moved
    def slider_dynamics_moved(self, choice):
        self.show_dynamics_help()

    # methods for various dropdown menus
    def time_signature_clicked(self, choice):
        self.display_help()
        CTkLabel(
            self.frame_help,
            text='Steps to Change Time Signature:',
            font=('Helvetica', 16, 'bold')
        ).grid(
            row=0, 
            column=0,
            padx=10,
            pady=10
        )
        CTkLabel(
            self.frame_help,
            text='1.Accessing the Time Signature Options:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=1, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='Locate the time signature on your score. This is usually at the beginning of the piece. Click on the current time signature or use the toolbar menu to open the time signature options.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=2, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='2.Understanding Time Signature Options:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=3, 
            column=0,
            padx=10,
            pady=(10,0)
        )
        CTkLabel(
            self.frame_help,
            text='The top number indicates how many beats are in each measure, and the bottom number shows which note value represents one beat.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=4, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Selecting a New Time Signature:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=5, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Scroll through the dropdown menu to find the time signature you wish to use.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=6, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='4.Applying the Selected Time Signature:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=7, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Click on your chosen time signature to select it. The new time signature will be applied to your score, starting from the current position or the beginning of the next measure.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=8, 
            column=0,
            padx=10,
        )

    def transpose_mode_clicked(self, choice):
        if choice == 'minor':
            self.dropdown_transpose.configure(
                values=self.minor_keys
            )
        elif choice == 'major':
            self.dropdown_transpose.configure(
                values=self.major_keys
            )
        self.var_transpose.set('')
        self.show_transpose_help()
    
    def transpose_clicked(self, choice):
        self.show_transpose_help()

    def show_transpose_help(self):
        self.display_help()
        CTkLabel(
            self.frame_help,
            text='Steps to Transpose :',
            font=('Helvetica', 16, 'bold')
        ).grid(
            row=0, 
            column=0,
            padx=10,
            pady=10
        )
        CTkLabel(
            self.frame_help,
            text='1.Understanding the Interface:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=1, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='The first dropdown menu displays the current key of your piece. The second dropdown menu allows you to choose between major and minor keys. The up/down arrows are used for making semitone adjustments.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=2, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='2.Selecting the Current Key:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=3, 
            column=0,
            padx=10,
            pady=(10,0)
        )
        CTkLabel(
            self.frame_help,
            text='Use the first dropdown menu to confirm the current key of your piece. If unsure, leave it as is; Crossfade will detect the key automatically.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=4, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Choosing Major or Minor:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=5, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text="Select whether the piece is in a major or minor key using the second dropdown menu.",
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=6, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='4.Transposing by Semitones:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=7, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Click the up arrow to transpose the music up by one semitone. Click the down arrow to transpose down by one semitone. Each click changes the key in the first dropdown menu accordingly.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=8, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='5.Confirming the Transposition:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=9, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='After adjusting, review the new key displayed in the first dropdown menu. Your music will automatically be transposed to this new key.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=10, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='6.Playback and Review:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=11, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='Use the playback feature to listen to your transposed music and ensure it sounds as expected.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=12, 
            column=0,
            padx=10,
        )


    def keys_clicked(self, choice):
        self.show_keys_help()

    def show_keys_help(self):
        self.display_help()
        CTkLabel(
            self.frame_help,
            text='Steps to Change Key Signature:',
            font=('Helvetica', 16, 'bold')
        ).grid(
            row=0, 
            column=0,
            padx=10,
            pady=10
        )
        CTkLabel(
            self.frame_help,
            text='1.Accessing Key Signature Options:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=1, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='Find the key signature in your score, usually located at the beginning of the piece. Click on the key signature or use the toolbar menu to open the key signature options ',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=2, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='2.Selecting a Key:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=3, 
            column=0,
            padx=10,
            pady=(10,0)
        )
        CTkLabel(
            self.frame_help,
            text='The first dropdown menu lists all available keys Scroll through and select the key that you wish to use for your composition.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=4, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='3.Choosing Major or Minor:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=5, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text="The second dropdown menu allows you to select 'Major' or 'Minor'. This choice determines the mood and tonal quality of your piece — major keys generally sound bright and happy, while minor keys often have a more somber tone.",
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=6, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='4.Applying the Key Signature:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=7, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='After selecting the key and choosing between major and minor, the new key signature will be applied to your score.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=8, 
            column=0,
            padx=10,
        )
        CTkLabel(
            self.frame_help,
            text='5.Reviewing the Score:',
            font=('Helvetica', 14, 'bold')
        ).grid(
            row=9, 
            column=0,
            padx=10,
            pady=(10,0),
        )
        CTkLabel(
            self.frame_help,
            text='After applying the new key signature, review your score to ensure the notes are correctly adjusted to the new key.',
            font=('Helvetica', 12),
            wraplength=self.help_wraplength
        ).grid(
            row=10, 
            column=0,
            padx=10,
        )


    def keys_mode_clicked(self, choice):
        if choice == 'minor':
            self.dropdown_keys.configure(
                values=self.minor_keys
            )
        elif choice == 'major':
            self.dropdown_keys.configure(
                values=self.major_keys
            )
        self.var_keys.set('')
        self.show_keys_help()

    # Method to maximize the window
    def maximize(self):
        self.state("zoomed")
        
    # Method to add the menu
    def add_menu(self):
        self.menu_bar = Menu(self)
        self.menu_file = Menu(self.menu_bar, tearoff=0)
        self.menu_edit = Menu(self.menu_bar, tearoff=0)
        self.menu_view = Menu(self.menu_bar, tearoff=0)
        self.menu_tools = Menu(self.menu_bar, tearoff=0)
        self.menu_help = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.menu_file)
        self.menu_bar.add_cascade(label="Edit", menu=self.menu_edit)
        self.menu_bar.add_cascade(label="View", menu=self.menu_view)
        self.menu_bar.add_cascade(label="Tools", menu=self.menu_tools)
        self.menu_bar.add_cascade(label="Help", menu=self.menu_help)
        self.submenu_theme = Menu(self.menu_view, tearoff=0)
        self.submenu_import = Menu(self.menu_file, tearoff=0)
        self.submenu_import.add_command(
            label='Import Audio', 
            command=self.import_audio
        )
        self.submenu_import.add_command(
            label='Import MIDI',
            command=self.import_midi
        )
        self.submenu_import.add_command(
            label='Import MusicXML', 
            command=self.import_musicxml
        )

        self.menu_file.add_cascade(
            label="Import File",
            menu=self.submenu_import
        )
        self.menu_file.add_separator()
        self.menu_file.add_command(
            label='Exit',
            command=self.destroy
        )
        self.submenu_theme.add_command(
            label='Light Mode', 
            command=lambda:self.change_mode('light')
        )
        self.submenu_theme.add_command(
            label='Dark Mode', 
            command=lambda:self.change_mode('dark')
        )
        self.menu_view.add_checkbutton(
            label="Time Signatures",
            variable=self.var_view_signatures,
            command=self.view_signatures
        )
        self.menu_view.add_checkbutton(
            label="Dynamics",
            variable=self.var_view_dynamics,
            command=self.view_dynamics
        )
        self.menu_view.add_checkbutton(
            label="Notes Duration",
            variable=self.var_view_duration,
            command=self.view_duration
        )
        self.menu_view.add_checkbutton(
            label="Change key",
            variable=self.var_view_key,
            command=self.view_key
        )
        self.menu_view.add_checkbutton(
            label="Transpose",
            variable=self.var_view_transpose,
            command=self.view_transpose
        )
        self.menu_view.add_separator()
        self.menu_view.add_cascade(
            label="Theme",
            menu=self.submenu_theme
        )
        self.config(menu=self.menu_bar)

    # Methods to handle visibility of various frames
    def view_transpose(self):
        if self.frame_transpose.winfo_ismapped():
            self.frame_transpose.grid_forget()
            self.configurator['view']['transpose'] = "0"
        else:
            self.frame_transpose.grid(
                row=5,
                column=0,
                sticky='NEW',
                padx=10,
                pady=10
            )
            self.configurator['view']['transpose'] = "1"
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    def view_key(self):
        if self.frame_key.winfo_ismapped():
            self.frame_key.grid_forget()
            self.configurator['view']['key'] = "0"
        else:
            self.frame_key.grid(
                row=4,
                column=0,
                sticky='NEW',
                padx=10,
                pady=10
            )
            self.configurator['view']['key'] = "1"
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    def view_duration(self):
        if self.frame_notes.winfo_ismapped():
            self.frame_notes.grid_forget()
            self.configurator['view']['duration'] = "0"
        else:
            self.frame_notes.grid(
                row=3,
                column=0,
                sticky='NEW',
                padx=10,
                pady=10
            )
            self.configurator['view']['duration'] = "1"
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    def view_dynamics(self):
        if self.frame_dynamics.winfo_ismapped():
            self.frame_dynamics.grid_forget()
            self.configurator['view']['dynamics'] = "0"
        else:
            self.frame_dynamics.grid(
                row=2,
                column=0,
                sticky='NEW',
                padx=10,
                pady=10
            )
            self.configurator['view']['dynamics'] = "1"
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    def view_signatures(self):
        if self.frame_signatures.winfo_ismapped():
            self.frame_signatures.grid_forget()
            self.configurator['view']['signatures'] = "0"
        else:
            self.frame_signatures.grid(
                row=0,
                column=0,
                sticky='NEW',
                padx=10,
                pady=10
            )
            self.configurator['view']['signatures'] = "1"
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    # Method to change the appearance mode (light or dark)
    def change_mode(self, mode):
        set_appearance_mode(mode)
        self.configurator['view']['theme'] = mode
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)

    # Methods to handle file imports
    def import_audio(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        if self.audio_path:
            print(f"Selected audio file: {self.audio_path}")
    
    def import_midi(self):
        self.midi_path = filedialog.askopenfilename(filetypes=[("MIDI Files", "*.mid;*.midi")])
        if self.midi_path:
            print(f"Selected MIDI file: {self.midi_path}")

    def import_musicxml(self):
        self.musicxml_path = filedialog.askopenfilename(filetypes=[("MusicXML Files", "*.mxl")])
        if self.musicxml_path:
            score = m21.converter.parse(self.musicxml_path)

            #Clear canvas
            for child in self.canvas_frame.winfo_children():
                child.pack_forget()

            # Save it as a PNG image
            img_path = score.write('musicxml.png', fp='Images/output.png')
            img = Image.open(img_path)
            img.thumbnail(
                (
                    self.frame_timeline.winfo_width(), 
                    10e99
                ), Image.ANTIALIAS
            )
            self.img_timeline = ImageTk.PhotoImage(img)
            Label(self.canvas_frame, image=self.img_timeline).pack()
            #self.canvas_frame.create_image(0, 0, image = self.img_timeline, anchor = NW, tags="timeline_image")
            #self.canvas_frame.bind("<Configure>",lambda e: self.canvas_timeline.configure(scrollregion=self.canvas_timeline.bbox("all")))
            #self.canvas_timeline.create_window((0, 0), window=self.canvas_frame, anchor="nw")
            #self.canvas_timeline.configure(height=self.img_timeline.height())

# Main entry point
if __name__ == '__main__':
    app = App()
    app.mainloop()
