import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from Video_Player import VideoPlayer
from VideoPlayerUI import VideoPlayerUI

def main():
    root = TkinterDnD.Tk()  # Use TkinterDnD for drag-and-drop support
    root.title("AccelAV Suite")

    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)

    control_frame = tk.Frame(root, bg="gray")
    player = VideoPlayer(canvas, control_frame)
    ui = VideoPlayerUI(root, player, control_frame)

    root.mainloop()

if __name__ == "__main__":
    main()
