import tkinter as tk
from tkinter import ttk, filedialog
import scipy.io.wavfile as wavfile
import tempfile, subprocess, os, threading, re, uuid
from TimeInputField import TimeInputField

class AudioExtractionWindow:
    def __init__(self, video_player):
        self.video_player = video_player
        self.window = tk.Toplevel()
        self.window.title("Extract Audio")
        self.format_var = tk.StringVar(value="WAV")
        self.format_options = ["WAV", "MP3", "FLAC"]
        self.create_widgets()

    def create_widgets(self):
        # --- Row 1: Time Input Fields ---
        self.time_frame = tk.Frame(self.window)
        tk.Label(self.time_frame, text="Start Time (HH:MM:SS:ms):").grid(row=0, column=0, padx=5, pady=5)
        self.start_time_widget = TimeInputField(self.time_frame, initial_time="00:00:00:000")
        self.start_time_widget.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.time_frame, text="End Time (HH:MM:SS:ms):").grid(row=0, column=2, padx=5, pady=5)
        if hasattr(self.video_player, 'audio_array_int') and self.video_player.audio_array_int is not None:
            full_duration = len(self.video_player.audio_array_int) / self.video_player.audio_sample_rate
            hours = int(full_duration // 3600)
            minutes = int((full_duration % 3600) // 60)
            seconds = int(full_duration % 60)
            milliseconds = int((full_duration - int(full_duration)) * 1000)
            default_end = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
            self.full_duration = full_duration
        else:
            default_end = "00:00:00:000"
            self.full_duration = 0
        
        self.end_time_widget = TimeInputField(self.time_frame, initial_time=default_end)
        self.end_time_widget.grid(row=0, column=3, padx=5, pady=5)
        self.time_frame.pack()

        if self.full_duration > 0:
            for var in [self.end_time_widget.hour_var,
                        self.end_time_widget.minute_var,
                        self.end_time_widget.second_var,
                        self.end_time_widget.millisecond_var]:
                var.trace_add('write', self.validate_end_time)
            
            for var in [self.start_time_widget.hour_var,
                        self.start_time_widget.minute_var,
                        self.start_time_widget.second_var,
                        self.start_time_widget.millisecond_var]:
                var.trace_add('write', self.validate_start_time)

        # --- Row 2: Buttons and Format Selection ---
        self.button_frame = tk.Frame(self.window)
        self.full_button = tk.Button(self.button_frame, text="Save Full Audio", command=self.extract_audio_full)
        self.full_button.pack(side="left", padx=5, pady=5)
        self.segment_button = tk.Button(self.button_frame, text="Extract Segment", command=self.extract_audio_segment)
        self.segment_button.pack(side="left", padx=5, pady=5)
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.exit_extraction)
        self.exit_button.pack(side="left", padx=5, pady=5)
        tk.Label(self.button_frame, text="Format:").pack(side="left", padx=5, pady=5)
        self.format_menu = tk.OptionMenu(self.button_frame, self.format_var, *self.format_options)
        self.format_menu.pack(side="left", padx=5, pady=5)
        self.button_frame.pack()

        # --- Row 3: Progress Section ---
        self.progress_frame = tk.Frame(self.window)
        self.progress_title = tk.Label(self.progress_frame, text="Extraction Progress:")
        self.progress_title.pack()
        self.progress = ttk.Progressbar(self.progress_frame, length=200, mode="determinate")
        self.progress.pack(side="left", padx=5)
        self.progress_label = tk.Label(self.progress_frame, text="0%")
        self.progress_label.pack(side="left")
        self.progress_frame.pack(pady=5)
        self.progress_frame.pack_forget()

    def validate_end_time(self, *args):
        """Ensure the end time never exceeds the video's total duration."""
        try:
            current_end = self.end_time_widget.get_time_in_seconds()
        except ValueError:
            return
        if current_end > self.full_duration:
            # Auto-correct to full duration.
            hours = int(self.full_duration // 3600)
            minutes = int((self.full_duration % 3600) // 60)
            seconds = int(self.full_duration % 60)
            milliseconds = int((self.full_duration - int(self.full_duration)) * 1000)
            self.end_time_widget.hour_var.set(f"{hours:02d}")
            self.end_time_widget.minute_var.set(f"{minutes:02d}")
            self.end_time_widget.second_var.set(f"{seconds:02d}")
            self.end_time_widget.millisecond_var.set(f"{milliseconds:03d}")

    def validate_start_time(self, *args):
        """Ensure the start time never exceeds the current end time."""
        try:
            start_time = self.start_time_widget.get_time_in_seconds()
            end_time = self.end_time_widget.get_time_in_seconds()
        except ValueError:
            return
        if start_time > end_time:
            # Auto-correct start time to match the current end time.
            hours = int(end_time // 3600)
            minutes = int((end_time % 3600) // 60)
            seconds = int(end_time % 60)
            milliseconds = int((end_time - int(end_time)) * 1000)
            self.start_time_widget.hour_var.set(f"{hours:02d}")
            self.start_time_widget.minute_var.set(f"{minutes:02d}")
            self.start_time_widget.second_var.set(f"{seconds:02d}")
            self.start_time_widget.millisecond_var.set(f"{milliseconds:03d}")

    def flash_message(self, message, duration=3000):
        """Display a flash message in the extraction window."""
        flash_label = tk.Label(self.window, text=message, bg="black", fg="white",
                               font=("Helvetica", 12, "bold"))
        flash_label.place(relx=0, rely=0, anchor="nw", x=10, y=10)
        flash_label.after(duration, flash_label.destroy)

    def exit_extraction(self):
        """Close the extraction window."""
        self.window.destroy()

    def extract_audio_full(self):
        threading.Thread(target=self._extract_audio_full_thread, daemon=True).start()

    def extract_audio_segment(self):
        threading.Thread(target=self._extract_audio_segment_thread, daemon=True).start()

    def _extract_audio_full_thread(self):
        output_format = self.format_var.get()
        ext_map = {"WAV": ".wav", "MP3": ".mp3", "FLAC": ".flac"}
        default_ext = ext_map.get(output_format, ".wav")

        # Generate a random file name suggestion.
        random_name = f"{uuid.uuid4().hex}{default_ext}"
        output_path = filedialog.asksaveasfilename(
            initialfile=random_name,
            defaultextension=default_ext,
            filetypes=[(f"{output_format} files", f"*{default_ext}")],
            title="Save Full Extracted Audio As"
        )
        if output_path:
            try:
                if not hasattr(self.video_player, 'audio_sample_rate') or self.video_player.audio_array_int is None:
                    raise ValueError("No audio data available for extraction.")

                self.window.after(0, self.show_progress)

                if output_format == "WAV":
                    wavfile.write(output_path, self.video_player.audio_sample_rate, self.video_player.audio_array_int)
                    self.window.after(0, self.complete_extraction)
                else:
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_wav_path = temp_wav.name
                    temp_wav.close()
                    wavfile.write(temp_wav_path, self.video_player.audio_sample_rate, self.video_player.audio_array_int)
                    total_duration = self.get_media_duration(temp_wav_path)
                    cmd = ['ffmpeg', '-y', '-i', temp_wav_path, output_path, '-nostats', '-progress', 'pipe:1']
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                               text=True, bufsize=1, universal_newlines=True)
                    for line in iter(process.stdout.readline, ''):
                        match = re.search(r'out_time_ms=(\d+)', line)
                        if match:
                            current_time = int(match.group(1)) / 1_000_000
                            progress = (current_time / total_duration) * 100
                            self.window.after(0, self.update_progress, progress)
                    process.wait()
                    os.unlink(temp_wav_path)
                    self.window.after(0, self.update_progress, 100)
                    self.window.after(0, self.complete_extraction)
            except Exception as e:
                self.window.after(0, lambda: self.flash_message(f"Full audio extraction failed:\n{str(e)}", 3000))

    def _extract_audio_segment_thread(self):
        output_format = self.format_var.get()
        ext_map = {"WAV": ".wav", "MP3": ".mp3", "FLAC": ".flac"}
        default_ext = ext_map.get(output_format, ".wav")

        # Generate a random file name suggestion.
        random_name = f"{uuid.uuid4().hex}{default_ext}"
        output_path = filedialog.asksaveasfilename(
            initialfile=random_name,
            defaultextension=default_ext,
            filetypes=[(f"{output_format} files", f"*{default_ext}")],
            title="Save Extracted Audio Segment As"
        )
        if output_path:
            try:
                if not hasattr(self.video_player, 'audio_sample_rate') or self.video_player.audio_array_int is None:
                    raise ValueError("No audio data available for extraction.")

                full_duration = len(self.video_player.audio_array_int) / self.video_player.audio_sample_rate

                start_time = self.start_time_widget.get_time_in_seconds()
                end_time = self.end_time_widget.get_time_in_seconds()

                if start_time < 0:
                    raise ValueError("Start time cannot be less than 0.")
                if end_time > full_duration:
                    raise ValueError("End time cannot exceed total audio duration.")
                if start_time >= end_time:
                    raise ValueError("Start time must be less than end time.")

                self.window.after(0, self.show_progress)
                segment_duration = end_time - start_time

                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                wavfile.write(temp_wav_path, self.video_player.audio_sample_rate, self.video_player.audio_array_int)

                cmd = ['ffmpeg', '-y', '-ss', str(start_time), '-to', str(end_time),
                       '-i', temp_wav_path, output_path, '-nostats', '-progress', 'pipe:1']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           text=True, bufsize=1, universal_newlines=True)
                for line in iter(process.stdout.readline, ''):
                    match = re.search(r'out_time_ms=(\d+)', line)
                    if match:
                        current_time = int(match.group(1)) / 1_000_000
                        adjusted_time = current_time - start_time
                        if adjusted_time < 0:
                            adjusted_time = 0
                        progress = (adjusted_time / segment_duration) * 100
                        self.window.after(0, self.update_progress, progress)
                process.wait()
                os.unlink(temp_wav_path)
                self.window.after(0, self.update_progress, 100)
                self.window.after(0, self.complete_extraction)
            except Exception as e:
                self.window.after(0, lambda: self.flash_message(f"Segment extraction failed:\n{str(e)}", 3000))

    def show_progress(self):
        self.progress["value"] = 0
        self.progress_label.config(text="0%")
        self.progress_frame.pack()

    def update_progress(self, value):
        self.progress["value"] = min(100, value)
        self.progress_label.config(text=f"{int(value)}%")

    def complete_extraction(self):
        self.progress["value"] = 100
        self.progress_label.config(text="100%")
        self.window.after(500, self.progress_frame.pack_forget)
        self.flash_message("Audio extraction completed!", 3000)

    def get_media_duration(self, file_path):
        """Retrieve media duration (in seconds) using ffprobe."""
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                   '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return float(result.stdout.strip())
        except Exception:
            return 10