import tkinter as tk
from tkinter import ttk, filedialog

class Editor:
    def __init__(self, master, main_menu_callback):
        self.master = master
        self.master.title("Editor Window")
        self.main_menu_callback = main_menu_callback  # Callback to main menu

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
        self.notebook.add(self.settings_frame, text="Tutorial")

        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Add buttons to the Editor frame
        self.import_button = tk.Button(self.main_menu_frame, text="Import File", command=self.on_import_button_click)
        self.import_button.pack()

        self.open_last_project_button = tk.Button(self.main_menu_frame, text="Open Last Project", command=self.open_last_project)
        self.open_last_project_button.pack()

        # Additional buttons for Profile and Settings
        self.profile_button = tk.Button(self.profile_frame, text="Profile Button", command=self.on_profile_button_click)
        self.profile_button.pack()

        self.settings_button = tk.Button(self.settings_frame, text="Settings Button", command=self.on_settings_button_click)
        self.settings_button.pack()

        # Main Menu tab in Editor
        self.notebook.bind("<Button-1>", self.on_tab_click)  # Bind left mouse click to the tab

    def on_tab_click(self, event):
        # Check if the click is on the "Main Menu" tab
        current_tab = self.notebook.tk.call(self.notebook._w, "identify", "tab", event.x, event.y)
        if "Main Menu" in current_tab:
            # If so, go back to the main menu
            self.master.destroy()
            self.main_menu_callback()

    def on_import_button_click(self):
        file_path = filedialog.askopenfilename(title="Select a file to import")
        if file_path:
            # Update the label or perform other actions as needed
            print("Imported file:", file_path)

    def open_last_project(self):
        # Add logic to open the last project
        print("Opening last project...")

    def on_profile_button_click(self):
        # Add your profile logic here
        print("Profile clicked")

    def on_settings_button_click(self):
        # Add your settings logic here
        print("Settings clicked")

# Callback function to go back to the main menu
def go_to_main_menu():
    root.deiconify()  # Show the main menu window

# Function to open the editor window
def open_editor_window():
    global editor_window
    editor_window = tk.Toplevel(root)
    editor = Editor(editor_window, go_to_main_menu)

# Create the main window
root = tk.Tk()
root.title("CrossFade Main Menu")

# Create widgets
label = tk.Label(root, text="Options")
settings_button = tk.Button(root, text="Settings", command=open_editor_window)
profile_button = tk.Button(root, text="Profile", command=open_editor_window)
editor_button = tk.Button(root, text="Editor", command=open_editor_window)  # Added Editor button