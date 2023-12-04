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
        file_path = filedialog.askopenfilename(title="Select a PDF file to import", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            # Display the PDF file on the canvas
            self.display_pdf(file_path)

    def display_pdf(self, file_path):
        try:
            pdf_document = fitz.open(file_path)
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image = page.get_pixmap()
                image = Image.frombytes("RGB", (image.width, image.height), image.samples)
                photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo  # Keep a reference to the image to prevent it from being garbage collected

            pdf_document.close()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
        except Exception as e:
            print(f"Error displaying PDF file: {e}")

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

# Organize widgets using grid layout
label.grid(row=0, column=0, padx=10, pady=10, columnspan=3)  # Increased columnspan to accommodate the Editor button
settings_button.grid(row=1, column=0, padx=10, pady=10)
profile_button.grid(row=1, column=1, padx=10, pady=10)
editor_button.grid(row=1, column=2, padx=10, pady=10)  # Placed the Editor button in the third column

# Start the main event loop
root.mainloop()
