import cv2
import pygame
import threading
import time
import tkinter as tk
from tkinter import PhotoImage
import torch
import numpy as np
from PIL import Image, ImageTk
from moviepy import VideoFileClip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
from UIControlManager import UIControlManager
from AudioVisualizer import AudioVisualizer
import tempfile, subprocess, os, random
import traceback

class VideoPlayer:
    # --------------------- Initialization & Setup ---------------------
    def __init__(self, canvas, control_frame):
        self.canvas = canvas
        self.control_frame = control_frame
        # Initialize UI Control Manager
        self.ui_controls = UIControlManager(self.canvas, self.control_frame)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cap = None
        self.video_path = None
        self.paused = False
        self.running = False  # True when playback is active
        self.fps = 30
        self.target_size = (canvas.winfo_width(), canvas.winfo_height())
        self.decoding_thread = None

        # GPU acceleration setup
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.stream = cv2.cuda_Stream() if self.use_cuda else None

        # Timing and synchronization
        self.playback_start_time = None
        self.playback_lock = threading.Lock()  # Lock for playback_start_time

        # Latest frame variables for sync/display
        self.latest_frame = None
        self.latest_frame_timestamp = 0.0

        self.audio_start_offset = 0.0

        # Audio configuration (processing on the CPU)
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        self.audio_sound = None
        self.audio_channel = None
        self.audio_sample_rate = None
        self.audio_array_int = None
        self.full_audio_array = None

        # Create a lock for VideoCapture operations
        self.cap_lock = threading.Lock()

        self.speed_multiplier = 1.0
        self.use_seeking_mode = False

        # For audio file play
        self.is_audio_file = False

        self.frame_lock = threading.Lock()

        self.seek_queue = []
        self.seek_lock = threading.Lock()
        self.seek_thread = None
        self.last_seek_time = 0

        # NEW: To track the scheduled after() callback for update_display
        self.update_display_id = None
        
        self.loop_mode = "no_loop"  # Loop mode can be "no_loop", "loop", "loop_current_file", or "shuffle"
        self.ended = False         # Flag to indicate if the file ended in no_loop mode
        self._end_handled = False   # Internal flag to prevent repeated end-of-file handling
        self.end_lock = threading.Lock()  # Lock for end-of-file handling

    # --------------------- Media Loading & Configuration ---------------------
    def load_video(self, video_path):
        self._end_handled = False
        self.video_path = video_path
        self.is_audio_file = False
        self._stop_playback()

        # Initialize audio visualizer reference
        self.audio_visualizer = None
        self.audio_data_normalized = None

        if video_path.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
            self.is_audio_file = True
            try:
                if video_path.lower().endswith('.wav'):
                    # Read WAV file using scipy.io.wavfile
                    self.audio_sample_rate, audio_data = wavfile.read(video_path)
                    
                    # If the file is floating point, assume its values are in [-1,1] and convert to int16.
                    if np.issubdtype(audio_data.dtype, np.floating):
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    # If the sample rate is not 44100, resample each channel individually.
                    if self.audio_sample_rate != 44100:
                        if audio_data.ndim > 1:
                            # For stereo (or multichannel), resample each channel.
                            resampled_channels = []
                            for ch in range(audio_data.shape[1]):
                                channel_float = audio_data[:, ch].astype(np.float32) / 32767.0
                                channel_resampled = librosa.resample(channel_float, orig_sr=self.audio_sample_rate, target_sr=44100)
                                resampled_channels.append(channel_resampled.astype(np.int16))
                            audio_data = np.column_stack(resampled_channels)
                        else:
                            channel_float = audio_data.astype(np.float32) / 32767.0
                            audio_data = librosa.resample(channel_float, orig_sr=self.audio_sample_rate, target_sr=44100)
                            audio_data = audio_data.astype(np.int16)
                        self.audio_sample_rate = 44100

                    self.audio_array_int = audio_data
                    self.audio_data_normalized = (audio_data[:, 0] if audio_data.ndim > 1 else audio_data).astype(np.float32) / 32767.0

                else:
                    self.audio_array_int, self.audio_sample_rate = librosa.load(
                        video_path,
                        sr=44100,
                        mono=True,
                        res_type='kaiser_fast'
                    )
                    self.audio_array_int = (self.audio_array_int * 32767).astype(np.int16)
                    self.audio_data_normalized = self.audio_array_int.astype(np.float32) / 32767.0

                if self.audio_array_int.ndim == 1:
                    self.audio_array_int = np.column_stack((self.audio_array_int, self.audio_array_int))

                self.audio_visualizer = AudioVisualizer(
                    data=self.audio_data_normalized,
                    sample_rate=self.audio_sample_rate,
                    chunk_size=256
                )

                self.audio_sound = pygame.sndarray.make_sound(self.audio_array_int)
                self.total_time = len(self.audio_array_int) / self.audio_sample_rate

            except Exception as e:
                print(f"Error loading audio file: {str(e)}")
                traceback.print_exc()
                return

            self.update_canvas_size(self.canvas.winfo_width(), self.canvas.winfo_height())
            return

        # Video file loading
        self.cap = cv2.VideoCapture(self.video_path)
        if self.use_cuda:
            self.cap.set(cv2.CAP_PROP_CUDA_DEVICE, 0)

        if not self.cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_time = total_frames / self.fps if self.fps else 0

        # Audio extraction using ffmpeg
        try:
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_name = temp_wav.name
            temp_wav.close()
            cmd = [
                'ffmpeg', '-y', '-i', self.video_path, '-vn',
                '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', temp_wav_name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            self.audio_sample_rate, audio_data = wavfile.read(temp_wav_name)
            os.unlink(temp_wav_name)
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            self.audio_array_int = audio_data
            self.audio_sound = pygame.sndarray.make_sound(self.audio_array_int)
            if audio_data.ndim > 1:
                self.audio_data_normalized = (audio_data[:, 0]).astype(np.float32) / 32767.0
            else:
                self.audio_data_normalized = audio_data.astype(np.float32) / 32767.0
        except Exception as video_error:
            print(f"Error loading video file: {video_error}")
            return

        self.update_canvas_size(self.canvas.winfo_width(), self.canvas.winfo_height())


    def update_canvas_size(self, new_width, new_height):
        if self.cap and self.cap.isOpened():
            original_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            original_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if original_width > 0 and original_height > 0:
                aspect_ratio = original_width / original_height
                canvas_ratio = new_width / new_height
                self.target_size = (
                    (new_width, int(new_width / aspect_ratio))
                    if aspect_ratio > canvas_ratio
                    else (int(new_height * aspect_ratio), new_height)
                )
                return
        self.target_size = (new_width, new_height)

    # --------------------- Playback Control ---------------------
    def play_video(self):
        # Check if there's either video or audio to play
        if not (self.cap or self.audio_sound):
            return

        # If already running and just paused, resume playback.
        if self.running:
            if self.paused:
                self.paused = False
                with self.playback_lock:
                    self.playback_start_time = time.perf_counter() - self.latest_frame_timestamp
                if self.audio_channel:
                    self.audio_channel.unpause()
            return

        self.paused = False
        self.running = True
        with self.playback_lock:
            self.playback_start_time = time.perf_counter()
        if self.audio_sound:
            self.audio_channel = self.audio_sound.play()

        # Only start decoding thread for video files
        if not self.is_audio_file:
            self.decoding_thread = threading.Thread(target=self.gpu_decoding_loop, daemon=True)
            self.decoding_thread.start()

        self.update_display()

    def pause_video(self):
        self.paused = True
        if self.audio_channel:
            self.audio_channel.pause()

    def _stop_playback(self):
        """Internal method to stop playback (used by load_video)"""
        self.running = False
        self.paused = False

        # Stop and reset audio
        if self.audio_channel:
            self.audio_channel.stop()
            self.audio_channel = None
        if hasattr(self, 'audio_sound'):
            self.audio_sound = None

        # Reset video capture
        if self.cap:
            with self.cap_lock:
                self.cap.release()
                self.cap = None

        # Clear frame buffers
        with self.frame_lock:
            self.latest_frame = None
            self.latest_frame_timestamp = 0.0

        # Reset audio array and clear pending audio data
        self.audio_array_int = None
        self.full_audio_array = None

        # Cancel any scheduled update_display callback
        if self.update_display_id is not None:
            self.canvas.after_cancel(self.update_display_id)
            self.update_display_id = None

        # Clear pending seek operations
        with self.seek_lock:
            self.seek_queue.clear()

        # Wait for decoding thread to finish (and join the seek thread)
        if self.decoding_thread and self.decoding_thread.is_alive():
            self.decoding_thread.join(timeout=0.5)
            self.decoding_thread = None

        if self.seek_thread and self.seek_thread.is_alive():
            self.seek_thread.join(timeout=0.5)
            self.seek_thread = None

        # Force a UI update after cleanup
        self.update_display(force=True)

    def handle_loop_end(self):
        if self.speed_multiplier > 0:
            new_video_time = 0.0
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            new_video_time = self.total_time
            with self.cap_lock:
                total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        with self.playback_lock:
            if self.speed_multiplier < 0:
                self.playback_start_time = time.perf_counter() - (self.total_time - new_video_time)/abs(self.speed_multiplier)
            else:
                self.playback_start_time = time.perf_counter() - new_video_time
        
        with self.frame_lock:
            self.latest_frame = None
            self.latest_frame_timestamp = new_video_time

    def handle_audio_end(self):
        self.running = False
        self.paused = False
        if self.audio_channel:
            self.audio_channel.stop()
        self.latest_frame_timestamp = self.total_time
        self.update_display()

    # --------------------- Video Processing & Decoding ---------------------
    def gpu_decoding_loop(self):
        while self.running:
            if self.cap is None:
                break

            if not self.paused:
                if self.use_seeking_mode:
                    with self.playback_lock:
                        current_real_time = time.perf_counter() - self.playback_start_time
                    current_video_time = current_real_time * self.speed_multiplier
                    current_video_time = max(0, min(current_video_time, self.total_time))
                    desired_frame = int(current_video_time * self.fps)

                    with self.cap_lock:
                        if self.cap is None:
                            break
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame)
                        ret, frame = self.cap.read()

                    if ret:
                        processed_frame = self.process_frame(frame)
                        actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                        actual_time = actual_pos / self.fps
                        with self.frame_lock:
                            self.latest_frame = processed_frame
                            self.latest_frame_timestamp = actual_time
                    else:
                        self.handle_end_of_file()
                    time.sleep(0.01)
                else:
                    with self.playback_lock:
                        current_time = time.perf_counter() - self.playback_start_time
                    with self.cap_lock:
                        if self.cap is None:
                            break
                        next_frame_index = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    next_frame_time = next_frame_index / self.fps

                    if current_time < next_frame_time:
                        time.sleep(next_frame_time - current_time)

                    with self.cap_lock:
                        if self.cap is None:
                            break
                        ret, frame = self.cap.read()
                        if ret:
                            actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                            actual_time = actual_pos / self.fps
                    if ret:
                        processed_frame = self.process_frame(frame)
                        with self.frame_lock:
                            self.latest_frame = processed_frame
                            self.latest_frame_timestamp = actual_time
                    else:
                        self.handle_end_of_file()
            else:
                time.sleep(0.01)

    def process_frame(self, frame):
        """
        Process the frame using the GPU (if available): convert to RGB and resize.
        """
        if self.use_cuda:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame, self.stream)
            gpu_mat = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2RGB)
            if self.target_size[0] > 0 and self.target_size[1] > 0:
                gpu_mat = cv2.cuda.resize(gpu_mat, self.target_size, interpolation=cv2.INTER_AREA)
            result = gpu_mat.download(stream=self.stream)
            self.stream.waitForCompletion()
            return result
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.target_size[0] > 0 and self.target_size[1] > 0:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            return frame

    # --------------------- Audio Processing & Visualization ---------------------
    def update_display(self, force=False):
        if not self.running and not force:
            if self.is_audio_file and not self.paused:
                self.canvas.delete("all")
            return

        if self.is_audio_file:
            if self.paused:
                current_time = self.latest_frame_timestamp
            else:
                current_real_time = time.perf_counter()
                with self.playback_lock:
                    playback_start = self.playback_start_time
                current_time = (current_real_time - playback_start) * self.speed_multiplier
                current_time = max(0, min(current_time, self.total_time))
            with self.frame_lock:
                self.latest_frame_timestamp = current_time

            # Check if end-of-file has been reached
            if (self.speed_multiplier > 0 and current_time >= self.total_time) or \
               (self.speed_multiplier < 0 and current_time <= 0):
                self.handle_end_of_file()

            if self.audio_visualizer and self.audio_data_normalized is not None:
                start_sample = int(current_time * self.audio_sample_rate)
                end_sample = start_sample + self.audio_visualizer.chunk_size

                if end_sample > len(self.audio_data_normalized):
                    end_sample = len(self.audio_data_normalized)
                    start_sample = max(0, end_sample - self.audio_visualizer.chunk_size)

                self.audio_visualizer.current_frame = start_sample // (self.audio_visualizer.chunk_size // 2)
                fig_image = self.audio_visualizer.render_frame()

                img = Image.fromarray(fig_image)
                photo = ImageTk.PhotoImage(
                    image=img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.Resampling.LANCZOS)
                )
                self.canvas.delete("all")
                self.canvas.create_image(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    anchor=tk.CENTER,
                    image=photo
                )
                self.canvas.image = photo
        else:
            with self.frame_lock:
                current_frame = self.latest_frame
            if current_frame is not None:
                photo = ImageTk.PhotoImage(image=Image.fromarray(current_frame))
                self.canvas.delete("all")
                self.canvas.create_image(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    anchor=tk.CENTER,
                    image=photo
                )
                self.canvas.image = photo

        if not force:
            self.update_display_id = self.canvas.after(5, self.update_display)

    # --------------------- Seeking Functionality ---------------------
    def seek(self, new_time, pause=False):
        """Thread-safe absolute seek"""
        with self.seek_lock:
            self.seek_queue.append(('absolute', new_time, pause))
        if self.seek_thread is None or not self.seek_thread.is_alive():
            self.seek_thread = threading.Thread(target=self._process_seek_queue, daemon=True)
            self.seek_thread.start()

    def _process_seek_queue(self):
        """Process accumulated seeks handling both types"""
        while True:
            with self.seek_lock:
                if not self.seek_queue:
                    return
                
                # Separate absolute and relative seeks
                absolute_seeks = [s for s in self.seek_queue if s[0] == 'absolute']
                relative_seeks = [s for s in self.seek_queue if s[0] == 'relative']
                
                # Clear queue after processing
                self.seek_queue.clear()

            # Process absolute seeks first (most recent one only)
            if absolute_seeks:
                # Use only the last absolute seek
                _, new_time, pause = absolute_seeks[-1]
                current_time = new_time
                self._perform_seek(new_time, pause)
            
            # Process relative seeks (sum all deltas)
            if relative_seeks:
                total_delta = sum(s[1] for s in relative_seeks)
                current_time = self.latest_frame_timestamp + total_delta
                current_time = max(0, min(current_time, self.total_time))
                self._perform_seek(current_time, pause=False)

            time.sleep(0.02)  # Reduced delay for better responsiveness
            self.canvas.event_generate("<<SeekUpdate>>", when="tail") 

    def _perform_seek(self, new_time, pause=False):
        """Actual seek implementation with synchronization fixes"""
        if not self.running and not pause:
            return

        new_time = max(0, min(new_time, self.total_time))
        
        # Unified timestamp for both audio and video
        target_time = new_time
        current_real_time = time.perf_counter()

        # Update playback timing first
        with self.playback_lock:
            self.playback_start_time = current_real_time - target_time
            self.latest_frame_timestamp = target_time

        # Video handling
        if not self.is_audio_file and self.cap and self.cap.isOpened():
            frame_index = int(target_time * self.fps)
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.cap.read()
                if ret:
                    processed_frame = self.process_frame(frame)
                    with self.frame_lock:
                        self.latest_frame = processed_frame
                        self.latest_frame_timestamp = target_time

        # Audio handling
        if self.audio_array_int is not None and not pause:
            # Stop any existing audio immediately
            if self.audio_channel:
                self.audio_channel.stop()
                
            # Calculate audio offset and restart
            audio_offset = int(target_time * self.audio_sample_rate)
            audio_offset = max(0, min(audio_offset, len(self.audio_array_int) - 1))
            sliced_audio = self.audio_array_int[audio_offset:]
            self.audio_sound = pygame.sndarray.make_sound(sliced_audio)
            self.audio_channel = self.audio_sound.play()

        # Schedule GUI update on main thread using event
        self.canvas.event_generate("<<SeekUpdate>>", when="tail")
        self.canvas.event_generate("<<SeekComplete>>")

    def seek_relative(self, delta):
        """Thread-safe relative seek"""
        with self.seek_lock:
            self.seek_queue.append(('relative', delta))
        if self.seek_thread is None or not self.seek_thread.is_alive():
            self.seek_thread = threading.Thread(target=self._process_seek_queue, daemon=True)
            self.seek_thread.start()

    # --------------------- Speed & Volume Control ---------------------
    def set_speed(self, speed):
        if speed == self.speed_multiplier:
            return

        # Get current state with locks
        with self.playback_lock, self.frame_lock:
            current_video_time = self.latest_frame_timestamp
            current_real_time = time.perf_counter()

        # Calculate new playback start time based on the current video time and speed
        new_speed = float(speed)
        if new_speed == 0:
            return

        new_start_time = current_real_time - (current_video_time / new_speed)

        # Update playback parameters with locks
        with self.playback_lock:
            self.playback_start_time = new_start_time
            self.speed_multiplier = new_speed
            self.use_seeking_mode = (new_speed != 1.0)

        # Handle audio synchronization
        if self.audio_channel:
            if abs(new_speed) != 1.0:
                # Pause audio and let video drive timing
                self.audio_channel.pause()
            else:
                # Restart audio at correct position for 1x speed
                if self.audio_array_int is not None:
                    offset = int(current_video_time * self.audio_sample_rate)
                    if offset < len(self.audio_array_int):
                        sliced_audio = self.audio_array_int[offset:]
                    else:
                        sliced_audio = self.audio_array_int[-1:]
                    
                    self.audio_sound = pygame.sndarray.make_sound(sliced_audio)
                    self.audio_channel = self.audio_sound.play()
                    # Sync audio to video timing
                    self.audio_channel.set_volume(self.audio_channel.get_volume())
                    
        # Force immediate frame update
        if not self.paused:
            with self.cap_lock:
                if self.cap and self.cap.isOpened():
                    frame_pos = int(current_video_time * self.fps)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = self.cap.read()
                    if ret:
                        processed_frame = self.process_frame(frame)
                        with self.frame_lock:
                            self.latest_frame = processed_frame
                            self.latest_frame_timestamp = current_video_time

    def set_volume(self, volume):
        """Set audio volume (0.0 to 1.0)"""
        if self.audio_channel:
            # Ensure volume is within valid range
            volume = max(0.0, min(1.0, float(volume)))
            self.audio_channel.set_volume(volume)


    def load_adjacent_video(self, direction="next"):
        """
        Load the adjacent video/audio file from the same directory.
        direction: "next" or "previous"
        """
        if not self.video_path:
            return

        # Define the supported extensions
        supported_extensions = (".mp4", ".avi", ".mkv", ".mp3", ".wav", ".m4a", ".ogg", ".flac")
        
        # Get the directory of the current video file
        current_dir = os.path.dirname(self.video_path)
        
        # List and sort all playable files in the directory
        all_files = os.listdir(current_dir)
        playable_files = sorted([
            f for f in all_files if f.lower().endswith(supported_extensions)
        ])

        current_file = os.path.basename(self.video_path)
        if current_file not in playable_files:
            print("Current file not found in directory listing.")
            return

        # Find the current file's index and compute the adjacent index (with wrap-around)
        idx = playable_files.index(current_file)
        if direction == "next":
            new_idx = (idx + 1) % len(playable_files)
        elif direction == "previous":
            new_idx = (idx - 1) % len(playable_files)
        else:
            return

        new_file = playable_files[new_idx]
        new_video_path = os.path.join(current_dir, new_file)
        print(f"Switching to: {new_video_path}")

        # Load and start playing the new file
        self.load_video(new_video_path)
        self.play_video()

    def load_next_video(self):
        if self.loop_mode == "shuffle":
            self.load_random_video()
        else:
            self.load_adjacent_video("next")

    def load_previous_video(self):
        if self.loop_mode == "shuffle":
            self.load_random_video()
        else:
            self.load_adjacent_video("previous")

    def handle_end_of_file(self):
        """
        Handle end-of-file events based on loop_mode.
        This method is executed only once per file termination.
        """
        with self.end_lock:
            if self._end_handled:
                return
            self._end_handled = True

        if self.loop_mode == "no_loop":
            self.running = False
            self.paused = True
            if self.audio_channel:
                self.audio_channel.stop()
            with self.playback_lock:
                self.latest_frame_timestamp = self.total_time
            self.ended = True
        elif self.loop_mode == "loop":
            # Schedule loading of the next video on the main thread.
            self.canvas.after(0, self.load_next_video)
        elif self.loop_mode == "loop_current_file":
            self._perform_seek(0, pause=False)
            # Reset the flag since we restarted the same file.
            with self.end_lock:
                self._end_handled = False
        elif self.loop_mode == "shuffle":
            self.canvas.after(0, self.load_random_video)

    def load_random_video(self):
        """
        Load a random video/audio file from the current directory that is different from the current file.
        """
        if not self.video_path:
            return
        supported_extensions = (".mp4", ".avi", ".mkv", ".mp3", ".wav", ".m4a", ".ogg", ".flac")
        current_dir = os.path.dirname(self.video_path)
        # Use the natural order from os.listdir rather than sorting.
        all_files = os.listdir(current_dir)
        playable_files = [f for f in all_files if f.lower().endswith(supported_extensions)]
        if not playable_files:
            return
        current_file = os.path.basename(self.video_path)
        if len(playable_files) > 1:
            playable_files = [f for f in playable_files if f != current_file]
        new_file = random.choice(playable_files)
        new_video_path = os.path.join(current_dir, new_file)
        print(f"Shuffle mode: Switching to random file: {new_video_path}")
        self.load_video(new_video_path)
        self.play_video()

    def open_audio_extraction_window(self):
        # Check if a file is loaded
        if not self.video_path:
            self.flash_message("Please load a video file before attempting audio extraction.")
            return

        # Check if the loaded file is an audio file
        if self.is_audio_file:
            self.flash_message("Audio extraction is only available for video files.")
            return

        # Otherwise, open the audio extraction window
        from AudioExtractionWindow import AudioExtractionWindow
        AudioExtractionWindow(self)

    def flash_message(self, message, duration=2000):
        """Display a flash message on the top left of the canvas for a few seconds."""
        label = tk.Label(self.canvas, text=message, bg="black", fg="white", font=("Helvetica", 12, "bold"))
        # Place the label at the top left (with a small offset)
        label.place(relx=0, rely=0, anchor="nw", x=10, y=10)
        label.after(duration, label.destroy)