import tkinter as tk
from tkinter import filedialog, messagebox
import scipy.io.wavfile as wavfile

class AudioExtractionWindow:
    def __init__(self, video_player):
        self.video_player = video_player  # Gives access to VideoPlayer variables
        self.window = tk.Toplevel()
        self.window.title("Extract Audio")
        self.create_widgets()

    def create_widgets(self):
        label = tk.Label(self.window, text="Extract audio from current media")
        label.pack(padx=10, pady=10)

        self.save_button = tk.Button(self.window, text="Save Extracted Audio", command=self.extract_audio)
        self.save_button.pack(pady=5)

        self.close_button = tk.Button(self.window, text="Close", command=self.window.destroy)
        self.close_button.pack(pady=5)

    def extract_audio(self):
        output_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            title="Save Extracted Audio As"
        )
        if output_path:
            try:
                # Access audio variables from the VideoPlayer instance
                if not hasattr(self.video_player, 'audio_sample_rate') or self.video_player.audio_array_int is None:
                    raise ValueError("No audio data available for extraction.")
                wavfile.write(output_path, self.video_player.audio_sample_rate, self.video_player.audio_array_int)
                messagebox.showinfo("Success", "Audio extraction completed!")
            except Exception as e:
                messagebox.showerror("Error", f"Audio extraction failed:\n{str(e)}")