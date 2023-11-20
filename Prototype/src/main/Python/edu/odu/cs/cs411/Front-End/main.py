from tkinter import *
from customtkinter import *
from PIL import Image

class App(CTk):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title('CrossFade Main Menu')
        self.add_menu()
        self.after(1, self.maximize)
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
        self.frame_settings = CTkFrame(self)
        self.frame_settings.grid(
            row=0,
            column=0,
            sticky='NSEW',
            padx=10,
            pady=10
        )
        self.frame_settings.grid_columnconfigure(0, weight=1)
        CTkLabel(
            self.frame_settings, 
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
            self.frame_settings,
            values=self.time_signatures,
            command=self.time_signature_clicked,
            variable=self.var_time_signatures
        ).grid(
            row=1,
            column=0,
            padx=5,
            pady=5
        )
        self.frame_main = CTkFrame(self)
        self.frame_main.grid(
            row=0,
            column=1,
            sticky='NSEW',
            padx=10,
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
        self.frame_dynamics.grid_columnconfigure(0, weight=1)
        CTkLabel(
            self.frame_dynamics, 
            text='Dynamics', 
            font=('Helvetica', 18, 'bold')
        ).grid(
            row=0,
            column=0,
            sticky='NEW',
            padx=5,
            pady=5
        )
        CTkSlider(
            self.frame_dynamics, 
            from_=0, 
            to=100, 
            command=self.slider1_moved
        ).grid(
            row=1,
            column=0,
            padx=10,
            pady=10
        )
        CTkSlider(
            self.frame_dynamics, 
            from_=0, 
            to=100, 
            command=self.slider2_moved
        ).grid(
            row=2,
            column=0,
            padx=10,
            pady=10
        )
        CTkSlider(
            self.frame_dynamics, 
            from_=0, 
            to=100, 
            command=self.slider3_moved
        ).grid(
            row=3,
            column=0,
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
            pady=5
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
            pady=5
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
            pady=5
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
            pady=5
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
            pady=5
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
            pady=5
        )
        CTkSlider(
            self.frame_notes, 
            from_=0, 
            to=100, 
            command=self.slidernote_moved
        ).grid(
            row=2,
            column=0,
            columnspan=6,
            padx=10,
            pady=10
        )
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)


    def slidernote_moved(self):
        pass


    def slider1_moved(self):
        pass


    def slider2_moved(self):
        pass


    def slider3_moved(self):
        pass


    def time_signature_clicked(self):
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
        self.submenu_import = Menu(self.menu_file, tearoff=0)
        self.submenu_import.add_command(
            label='Import Audio', 
            command=self.import_audio
        )
        self.submenu_import.add_command(
            label='Import MIDI',
            command=self.import_midi
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

        self.config(menu=self.menu_bar)


    def import_audio(self):
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        if self.audio_path:
            print(f"Selected audio file: {self.audio_path}")

    
    def import_midi(self):
        self.midi_path = filedialog.askopenfilename(filetypes=[("MIDI Files", "*.mid;*.midi")])
        if self.midi_path:
            print(f"Selected MIDI file: {self.midi_path}")


if __name__ == '__main__':
    app = App()
    app.mainloop()