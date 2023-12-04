import tkinter as tk
from tkinter import ttk, filedialog

# Define shades of black and blue
black = "#000000"
dark_blue = "#001F3F"
white = "#FFFFFF"

class Editor:
    def __init__(self, master, main_menu_callback):
        self.master = master
        self.master.title("Editor Window")

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(master)

        # Create frames for each tab
        self.main_menu_frame = tk.Frame(self.notebook, bg=black)

        # Add frames to the notebook with corresponding tab names
        self.notebook.add(self.main_menu_frame, text="Main Menu")

        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Import button in Editor frame
        self.import_button = tk.Button(self.main_menu_frame, text="Import File", command=self.on_import_button_click, bg=dark_blue, fg=white)
        self.import_button.pack(pady=10)

        # Configure row and column weights to make them expand with the window
        for i in range(5):  # Number of rows in the layout
            self.main_menu_frame.grid_rowconfigure(i, weight=1)
            self.main_menu_frame.grid_columnconfigure(i, weight=1)

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

# Callback function to go back to the main menu
def go_to_main_menu(second_window):
    second_window.destroy()  # Close the second window
    root.deiconify()  # Show the main menu window

# Function to open the editor window
def open_editor_window():
    # Hide the main menu window
    root.iconify()

    # Import the other script
    from main import App as OtherApp

    # Create a new Tk instance for the second window
    second_window = tk.Toplevel()
    second_window.title("Editor Window")

    # Create the App instance from the other script
    editor = OtherApp(second_window)

    # Start the second window's event loop
    second_window.mainloop()

# Create the main window
root = tk.Tk()
root.title("CrossFade Main Menu")
root.configure(bg=black)

# Main Menu label above the logo
label = tk.Label(root, text="Main Menu", font=("Arial", 14, "bold"), bg=black, fg=white)
label.grid(row=1, column=1, pady=10)

# Load and display a resized image on the Main Menu page
image_path = r"C:\Users\hyaci\OneDrive\Pictures\CF.png"  # Use a raw string
original_image = tk.PhotoImage(file=image_path)
resized_image = original_image.subsample(2, 2)  # Adjust the subsample values for resizing
image_label = tk.Label(root, image=resized_image, bg=black)
image_label.grid(row=2, column=1, pady=10, sticky="nsew")  # Span two rows for the image

# Create widgets
settings_button = tk.Button(root, text="Settings", command=open_editor_window, padx=20, pady=10, bg=dark_blue, fg=white, font=("Arial", 12, "bold"))
profile_button = tk.Button(root, text="Profile", command=open_editor_window, padx=20, pady=10, bg=dark_blue, fg=white, font=("Arial", 12, "bold"))
editor_button = tk.Button(root, text="Editor", command=open_editor_window, padx=20, pady=10, bg=dark_blue, fg=white, font=("Arial", 12, "bold"))  # Added Editor button

# Organize widgets using grid layout
settings_button.grid(row=3, column=0, pady=10)
profile_button.grid(row=3, column=2, pady=10)
editor_button.grid(row=4, column=1, pady=10)  # Placed the Editor button in the third column

# Configure row and column weights to make them expand with the window
for i in range(5):  # Number of rows in the layout
    root.grid_rowconfigure(i, weight=1)
    root.grid_columnconfigure(i, weight=1)

# Start the main event loop
root.mainloop()























