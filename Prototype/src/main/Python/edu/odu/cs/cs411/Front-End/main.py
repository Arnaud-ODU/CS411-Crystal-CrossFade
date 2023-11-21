from tkinter import *
from customtkinter import *
from PIL import Image
from configparser import ConfigParser
import os

class App(CTk):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.var_view_signatures = IntVar(value=1)
        self.var_view_dynamics = IntVar(value=1)
        self.var_view_duration = IntVar(value=1)
        self.configurator = ConfigParser()
        self.title('CrossFade Main Menu')
        self.add_menu()
        self.time_signatures = (
            '2/4',
            '3/4',
            '4/4'
        )
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
        self.add_settings()
        self.read_config()
        self.add_toolbar()
        self.add_timeline()
        self.after(1, self.maximize)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)


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


    def add_timeline(self):
        self.frame_timeline = CTkFrame(self)
        self.frame_timeline.grid(
            row=1,
            column=1,
            pady=(5,10),
            padx=(0,10),
            sticky='NSEW',
        )


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
        else:
            self.configurator['view'] = {
                'theme': 'dark',
                'signatures': 1,
                'dynamics': 1,
                'duration': 1
            }
 
 
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
        self.var_time_signatures = StringVar()
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
            width=1
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
            width=1
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
            width=1
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
            width=1
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
            width=1
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
            width=1
        )
        self.button_thirtysecondnote.grid(
            row=1,
            column=5,
            pady=10
        )


    def slider_dynamics_moved(self):
        pass


    def time_signature_clicked(self, choice):
        pass


    def maximize(self):
        self.state("zoomed")
        

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
        self.menu_view.add_separator()
        self.menu_view.add_cascade(
            label="Theme",
            menu=self.submenu_theme
        )
        self.config(menu=self.menu_bar)


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

    def change_mode(self, mode):
        set_appearance_mode(mode)
        self.configurator['view']['theme'] = mode
        with open('config.ini', 'w') as configfile:
            self.configurator.write(configfile)


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
            print(f"Selected MusicXML file: {self.musicxml_path}")


if __name__ == '__main__':
    app = App()
    app.mainloop()