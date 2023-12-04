import tkinter as tk
from tkinter import ttk, filedialog
import fitz  # PyMuPDF
from PIL import Image, ImageTk

class Editor:
    def __init__(self, master, main_menu_callback):
        self.master = master
        self.master.title("Editor Window")
        self.main_menu_callback = main_menu_callback  # Callback to the main menu

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(master)

        # Create frames for each tab
        self.main_menu_frame = tk.Frame(self.notebook)
        self.profile_frame = tk.Frame(self.notebook)
        self.settings_frame = tk.Frame(self.notebook)

        # Add frames to the notebook with corresponding tab names
        self.notebook.add(self.main_menu_frame, text="Main Menu")
        self.notebook.add(self.profile_frame, text="Profile")
        self.notebook.add(self.settings_frame, text="Settings")

        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add buttons to the Editor frame
        self.import_button = tk.Button(self.main_menu_frame, text="Import File", command=self.on_import_button_click)
        self.import_button.pack()

        self.open_last_project_button = tk.Button(self.main_menu_frame, text="Open Last Project", command=self.open_last_project)
        self.open_last_project_button.pack()

        # Canvas widget to display PDF file content
        self.canvas = tk.Canvas(self.main_menu_frame, width=600, height=800)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        self.scrollbar = tk.Scrollbar(self.main_menu_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Additional buttons for Profile and Settings
        self.profile_button = tk.Button(self.profile_frame, text="Profile Button", command=self.on_profile_button_click)
        self.profile_button.pack()

        self.settings_button = tk.Button(self.settings_frame, text="Settings Button", command=self.on_settings_button_click)
        self.settings_button.pack()
