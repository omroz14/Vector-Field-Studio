"""Vector Field Studio entry point."""
import sys

try:
    import tkinter  # noqa: F401
except ImportError:
    sys.exit("Tkinter is not installed.\n"
             "  Debian / Ubuntu / Linux Mint:  sudo apt install python3-tk\n"
             "  Windows / macOS: reinstall Python from python.org with the 'tcl/tk' option enabled.")

try:
    import customtkinter  # noqa: F401
except ImportError:
    sys.exit("CustomTkinter is not installed. Run:  pip install -r requirements.txt")

from ui import App

if __name__ == "__main__":
    App().mainloop()
