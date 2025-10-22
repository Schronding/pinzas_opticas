import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from interfaz import gui

if __name__ == "__main__":
    app = gui.App()
    app.mainloop()