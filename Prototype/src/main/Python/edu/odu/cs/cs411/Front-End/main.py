# Import necessary modules
from tkinter import *
from tkinter.filedialog import asksaveasfile, askdirectory, asksaveasfilename
from tkinter.tix import Select
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
from Backend.Song import *

# Define the main application class
class App(CTk):

    # Constructor for the application
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize various variables and settings
        self.song = Song() #Default Song To Store Unit A Song Is Imported
        self.imported = False #False Until A Song Has Been Imported
        self.var_view_signatures = IntVar(value=1)
        self.var_view_dynamics = IntVar(value=1)
        self.var_view_duration = IntVar(value=1)
        self.var_view_key = IntVar(value=1)
        self.var_view_transpose = IntVar(value=1)
        self.var_view_note_select = IntVar(value=1)
        self.configurator = ConfigParser()
        self.title('CrossFade Main Menu')

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
        #Note Durations
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

        #Top Of Player Task Bar
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
        self.after(1, self.maximize)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

        

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
            self.var_view_note_select.set(self.configurator['view']['select'])
            if not self.var_view_note_select.get():
                self.frame_note_select.grid_forget()
        else:
            self.configurator['view'] = {
                'theme': 'dark',
                'signatures': 1,
                'dynamics': 1,
                'duration': 1
            }
 
    # Method to add various settings
    def add_settings(self):

        #--------------------------------Settings Grouping-----------------------#
        self.frame_settings = CTkScrollableFrame(self, fg_color="transparent")
        self.frame_settings.grid(
            row=0,
            rowspan=2,
            column=0,
            sticky='NSEW',
            padx=10,
            pady=10
        )
        self.frame_settings.grid_columnconfigure(0, weight=1)
        
        #--------------------------------Time Signature Grouping-----------------#
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
       
        #--------------------------------Dynamics Grouping-----------------------#
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
            width=45
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
            width=45
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
            width=45
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
            width=45
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
            width=45
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
            width=45
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
        
        #-------------------------------Duration Grouping--------------------------#
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
            command=self.whole_clicked
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
            command=self.half_clicked
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
            command=self.quarter_clicked
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
            command=self.eighth_clicked
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
            command=self.sixteenth_clicked
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
            command=self.thirtysecond_clicked
        )#MARKER
        self.button_thirtysecondnote.grid(
            row=1,
            column=5,
            pady=10
        )
        
        #------------------------------Pitch Grouping----------------------------#
        self.frame_pitch = CTkFrame(
            self.frame_settings,
        )
        self.frame_pitch.grid(
            row=4,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
       
        self.frame_pitch.grid_columnconfigure((0,1,2), weight=1)
        CTkLabel(
            self.frame_pitch, 
            text='Notes Pitch', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky='NEW',
            padx=5,
            pady=5
        )

        self.frame_pitch_buttons = CTkFrame(
            self.frame_pitch,
        )
        self.frame_pitch_buttons.grid(
            row=1,
            column=3,
            padx=5,
            pady=10
        )


        self.octive_entry = CTkEntry(
            self.frame_pitch,    
            placeholder_text='Octive',
            placeholder_text_color='grey',
            width=80,
            height=10,
            fg_color='dark blue',
            justify='center'
        )
        self.octive_entry.grid(
            row=1,
            column=0
        )
       
        self.letter_entry = CTkEntry(
            self.frame_pitch,    
            placeholder_text='Note Name',
            placeholder_text_color='grey',
            width=80,
            height=10,
            fg_color='dark blue',
            justify='center'
        )
        self.letter_entry.grid(
            row=1,
            column=2
        )

        CTkButton(
            self.frame_pitch_buttons,
            text='Change Pitch',
            font=('Helvetica', 18),
            width=0,
            command=self.change_pitch_clicked
        ).grid(
            row=3,
            column=0
        )
        #------------------------------Key Grouping------------------------------#
        self.frame_key = CTkFrame(
            self.frame_settings,
        )
        self.frame_key.grid(
            row=5,
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
        
        #----------------------------Transpose Grouping---------------------------#
        self.frame_transpose = CTkFrame(
            self.frame_settings,
        )
        self.frame_transpose.grid(
            row=6,
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

        #----------------------------Note Select Grouping---------------------------#
        self.frame_note_select = CTkFrame(
            self.frame_settings,
        )
        self.frame_note_select.grid(
            row=7,
            column=0,
            sticky='NEW',
            padx=10,
            pady=10
        )
        
        self.frame_note_select.grid_columnconfigure((0,1,2), weight=1)
        CTkLabel(
            self.frame_note_select, 
            text='Note Selection', 
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
            self.frame_note_select, 
            text='(select the note to be adjusted)', 
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
        #CTkLabel(
        #    self.frame_note_select, 
        #    text='        Line', 
        #    font=('Helvetica', 12, 'bold'),
        #    justify = 'center',
        #    anchor="w"
        #).grid(
        #    row=2,
        #    column=0,
        #    columnspan=3,
        #    sticky='NEW',
        #    padx=5,
        #    pady=(5,0)
        #)
        #CTkLabel(
        #    self.frame_note_select, 
        #    text='Measure', 
        #    font=('Helvetica', 12, 'bold'),
        #    justify='center',
        #    anchor='w'
        #).grid(
        #    row=2,
        #    column=1,
        #    columnspan=3,
        #    sticky='NEW',
        #    padx=5,
        #    pady=(5,0)
        #)
        #CTkLabel(
        #    self.frame_note_select, 
        #    text='Note', 
        #    font=('Helvetica', 12, 'bold'),
        #    justify='center',
        #    anchor='w'
        #).grid(
        #    row=2,
        #    column=2,
        #    columnspan=3,
        #    sticky='NEW',
        #    padx=0,
        #    pady=(5,0)
        #)

        self.frame_note_select_buttons = CTkFrame(
            self.frame_note_select,
        )
        self.frame_note_select_buttons.grid(
            row=3,
            column=3,
            padx=5,
            pady=10
        )


        self.place_entry = CTkEntry(
            self.frame_note_select,    
            placeholder_text='Part',
            placeholder_text_color='grey',
            width=80,
            height=10,
            fg_color='dark blue',
            justify='center'
        )
        self.place_entry.grid(
            row=3,
            column=0
        )

        self.measure_entry = CTkEntry(
            self.frame_note_select,    
            placeholder_text='Measure',
            placeholder_text_color='grey',
            width=80,
            height=10,
            fg_color='dark blue',
            justify='center'
        )
        self.measure_entry.grid(
            row=3,
            column=1
        )

        self.note_entry = CTkEntry(
            self.frame_note_select,    
            placeholder_text='Note',
            placeholder_text_color='grey',
            width=80,
            height=10,
            fg_color='dark blue',
            justify='center'
        )
        self.note_entry.grid(
            row=3,
            column=2
        )
        
        self.note_info = CTkLabel(
            self.frame_note_select, 
            text='Note: None', 
            font=('Helvetica', 16, 'bold'),
            justify = 'center',
           anchor="w"
        )
        
        self.note_info.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky='NEW',
            padx=5,
            pady=(5,0)
        )

        self.duration_info = CTkLabel(
            self.frame_note_select, 
            text='Duration Type: None', 
            font=('Helvetica', 16, 'bold'),
            justify = 'center',
           anchor="w"
        )
        
        self.duration_info.grid(
            row=4,
            column=1,
            columnspan=3,
            sticky='NEW',
            padx=5,
            pady=(5,0)
        )

        CTkButton(
            self.frame_note_select_buttons,
            text='Select',
            font=('Helvetica', 18),
            width=0,
            command=self.get_selected_note
        ).grid(
            row=3,
            column=0
        )

        

        #self.place_textbox = Entry(width=30).place(x=50, y=290)
        #MARKER
  
    # Methods to handle increment and decrement of transpose
    def increment_transpose(self):
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
    def slider_dynamics_moved(self):
        pass

    # methods for various dropdown menus
    def time_signature_clicked(self, choice):
        pass

    def whole_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='whole')

    def half_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='half')

    def quarter_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='quarter')

    def eighth_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='eighth')
    
    def sixteenth_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='16th')
    
    def thirtysecond_clicked(self):
        part_num, measure_num, note_num = self.get_selected_note()
        self.change_duration(int(part_num), int(measure_num), int(note_num), length='32nd')

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
    
    def transpose_clicked(self, choice):
        pass

    def keys_clicked(self, choice):
        pass

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

    def change_pitch_clicked(self, part_num=-1, measure_num=-1, note_num=-1, new_pitch='C4'):
        """Change A Note In The Parsed Music

        Args:
            part_num (_int_): The number of the part where the note is located, where the first part would be 1
            measure_num (_int_): The number of the measure where the note is located
            note_num (_int_): The number of the note. If there are 5 notes, and the 3rd one must be changed, this number would be 3
            new_pitch (_string_): The New Pitch That The Note Needs To Have (ex: 'C#4')
        """
        part_num, measure_num, note_num = self.get_selected_note()
        new_pitch = self.letter_entry.get() + self.octive_entry.get()
        if part_num is not -1 and measure_num is not -1 and note_num is not -1:
            self.song.change_note_pitch(int(part_num), int(measure_num), int(note_num), new_pitch)
            self.get_selected_note() #Reselects The Modified Note (Avoids Null Pointer Exception)
            self.display()

    # Method to maximize the window
    def maximize(self):
        """Maximizes The Window"""
        self.state("zoomed")

    def change_duration(self, part_num=-1, measure_num=-1, note_num=-1, length='whole', num_dots=0):
        """"Changes The Duration Of A Specified Note To A Passed Duration
        
        Args:
            part_num (_int_): Index Of The Part That Contains The Note In song.parts
            measure_num (_int_): Index Of The Measure That Contains The Note In song.parts[i].measure
            note_num (_int_): Index Of The Note In song.part[i].measure(j).notes       
            length (_str_): String That Cooresponds To Desired Length In Music21 Library
            num_dots (_int_): The Number Of Dots That Should Be Listed On The Note ( Each Dot Adds .5 Duration To A Note)
        """
        if part_num is not -1 and measure_num is not -1 and note_num is not -1:
            self.song.change_duration(part_num, measure_num, note_num, length, num_dots)
            self.get_selected_note() #Reselects The Modified Note (Avoids Null Pointer Exception)
            self.display()

    def display_note(self, part_num=1, measure_num=1, note_num=1):
        """Checks I The Passed Note Is Valid, If Valid Displays Note Information
        
        Args:
            part_num (_int_): Index Of The Part That Contains The Note In song.parts
            measure_num (_int_): Index Of The Measure That Contains The Note In song.parts[i].measure
            note_num (_int_): Index Of The Note In song.part[i].measure(j).notes        
        """

        if not (self.song.parsed_music == None):
            try:
                if 'rest' not in self.song.parsed_music.parts[int(part_num)-1].measure(int(measure_num)).notesAndRests[int(note_num)-1].name:
                    self.note_info.configure(text='Note: ' + str(self.song.parsed_music.parts[int(part_num)-1].measure(int(measure_num)).notesAndRests[int(note_num)-1].pitch))
                else:
                    self.note_info.configure(text='Note: ' + str(self.song.parsed_music.parts[int(part_num)-1].measure(int(measure_num)).notesAndRests[int(note_num)-1].name))
                self.duration_info.configure(text='Duration: ' + self.song.parsed_music.parts[int(part_num)-1].measure(int(measure_num)).notesAndRests[int(note_num)-1].duration.type)
            except IndexError as e:
                pass

    def get_selected_note(self, internal_call=False, part_num=-1, measure_num=-1, note_num=-1):
        """Reads In The Part, Measure, And Note values (_int_) From The Textboxes
        
        Args:
            internal_call (_bool_): Bool Variable For Niche Situations And Testing
            part_num (_int_): Index Of The Part That Contains The Note In song.parts
            measure_num (_int_): Index Of The Measure That Contains The Note In song.parts[i].measure
            note_num (_int_): Index Of The Note In song.part[i].measure(j).notes
        """
        if not internal_call:
            part_num = self.place_entry.get()
            measure_num = self.measure_entry.get()
            note_num = self.note_entry.get()

        self.display_note(part_num, measure_num, note_num)
        return part_num, measure_num, note_num



        
    # Method to add the menu
    def add_menu(self):
        """Adds Menu Elements To The App"""
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

        self.submenu_export = Menu(self.menu_file, tearoff=0)
        self.submenu_export.add_command(
            label='Export Audio', 
            command=self.export_audio
        )
        self.submenu_export.add_command(
            label='Export MIDI',
            command=self.export_midi
        )
        self.submenu_export.add_command(
            label='Export MusicXML', 
            command=self.export_musicxml
        )
        self.menu_file.add_cascade(
            label="Export File",
            menu=self.submenu_export
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
            self.imported = True
    
    def import_midi(self):
        self.midi_path = filedialog.askopenfilename(filetypes=[("MIDI Files", "*.mid;*.midi")])
        if self.midi_path:
            print(f"Selected MIDI file: {self.midi_path}")
            self.imported = True

    def import_musicxml(self):
        self.musicxml_path = filedialog.askopenfilename(filetypes=[("MusicXML Files", "*.mxl;*.musicxml")])
        if self.musicxml_path:
            

            #Stores The Song After Importing
            if not self.imported:
                self.song.import_musicxml(self.musicxml_path)
            else:
                self.song = Song(self.musicxml_path)
            self.display()

            
            #self.canvas_frame.create_image(0, 0, image = self.img_timeline, anchor = NW, tags="timeline_image")
            #self.canvas_frame.bind("<Configure>",lambda e: self.canvas_timeline.configure(scrollregion=self.canvas_timeline.bbox("all")))
            #self.canvas_timeline.create_window((0, 0), window=self.canvas_frame, anchor="nw")
            #self.canvas_timeline.configure(height=self.img_timeline.height())
            self.imported = True


    def export_audio(self):
        pass


    def export_midi(self):
        if self.imported:
            self.song.export_midi(asksaveasfilename(filetypes=[("MIDI Files", "*.midi")]))

    def export_musicxml(self):
        if self.imported:
            self.song.export_musicxml(asksaveasfilename(filetypes=[("MusicXML Files", "*.musicxml")]))


    def display(self):
        """Displays the Current Version Of The MusicXML File"""
        score = self.song.parsed_music
        # Save it as a PNG image
        #Clear canvas
        for child in self.canvas_frame.winfo_children():
            child.pack_forget()

        img_path = score.write('musicxml.png', fp='Images/output.png')
        img = Image.open(img_path)
        img.thumbnail(
            (
                self.frame_timeline.winfo_width(), 
                10e99
            ), Image.Resampling.LANCZOS
        )
        self.img_timeline = ImageTk.PhotoImage(img)
        Label(self.canvas_frame, image=self.img_timeline).pack()
        img.close()

    def run(self):
        self.mainloop()


# Main entry point
if __name__ == '__main__':
    app = App()
    app.run()
    #app.mainloop()