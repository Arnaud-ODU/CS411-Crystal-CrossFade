import tkinter as tk

def on_button_click():
    label.config(text="Hello, " + entry.get())

# Create the main window
root = tk.Tk()
root.title("My GUI")

# Create widgets
label = tk.Label(root, text="Enter your name:")
entry = tk.Entry(root)
button = tk.Button(root, text="Say Hello", command=on_button_click)

# Organize widgets using grid layout
label.grid(row=0, column=0, padx=10, pady=10)
entry.grid(row=0, column=1, padx=10, pady=10)
button.grid(row=1, column=0, columnspan=2, pady=10)
