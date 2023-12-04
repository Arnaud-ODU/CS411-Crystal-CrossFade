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