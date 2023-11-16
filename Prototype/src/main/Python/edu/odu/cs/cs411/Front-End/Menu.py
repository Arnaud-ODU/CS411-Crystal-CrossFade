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
