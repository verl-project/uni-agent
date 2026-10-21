"""Tiny Tk app for GUI smoke tests: clicking the button writes a marker file.

Usage: button_app.py [marker_path]   (default /tmp/gui_button_clicked)
"""

import sys
import tkinter as tk

marker = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gui_button_clicked"

root = tk.Tk()
root.title("Button test")
root.geometry("400x300+100+100")


def on_click():
    with open(marker, "w") as f:
        f.write("clicked")
    status.config(text="Done!")


status = tk.Label(root, text="Click the button", font=("Helvetica", 18))
status.pack(pady=30)
tk.Button(root, text="SUBMIT", font=("Helvetica", 24), width=10, height=2, command=on_click).pack(pady=20)
root.mainloop()
