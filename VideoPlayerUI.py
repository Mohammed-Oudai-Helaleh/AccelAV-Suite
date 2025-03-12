import tkinter as tk
from tkinter import filedialog, ttk
from tkinterdnd2 import DND_FILES
from PIL import Image, ImageTk

class VideoPlayerUI:
    # --------------------- Initialization & UI Setup ---------------------
    def __init__(self, root, player, control_frame):
        self.root = root
        self.player = player
        self.control_frame = control_frame
        self.canvas = player.canvas
        self.root.geometry("800x600")

        # Volume control variables
        self.volume = 1.0  # Default volume (100%)
        self.muted = False
        self.pre_mute_volume = self.volume
        self.slider_active = False

        # Drag and Drop support
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

        # Progress components
        self.progress_frame = tk.Frame(self.control_frame, bg="gray", height=2)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.progress_bar.bind("<Button-1>", self.seek_video)
        self.progress_bar.bind("<B1-Motion>", self.scrub_video)
        self.progress_bar.bind("<ButtonRelease-1>", self.resume_video)
        
        self.time_label = tk.Label(self.progress_frame, text="00:00 / 00:00", bg="gray", fg="white")
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # Control buttons frame (full width container)
        self.buttons_frame = tk.Frame(self.control_frame, bg="gray", height=20)
        self.buttons_frame.pack(fill=tk.X, padx=2, pady=2)

        # --------------------- Create Group Frames ---------------------
        # Left group: will be packed to the left
        self.left_frame = tk.Frame(self.buttons_frame, bg="gray")
        self.left_frame.pack(side=tk.LEFT, padx=5)
        
        # Right group: will be packed to the right
        self.right_frame = tk.Frame(self.buttons_frame, bg="gray")
        self.right_frame.pack(side=tk.RIGHT, padx=5)
        
        # Middle group: will be placed at the center of buttons_frame
        self.middle_frame = tk.Frame(self.buttons_frame, bg="gray")
        self.middle_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # --------------------- Icon Setup ---------------------
        # Standard icons for playback controls
        self.play_icon = self.load_icon("./icons/play_icon.png", (24, 24))
        self.pause_icon = self.load_icon("./icons/pause_icon.png", (24, 24))
        self.load_icon_img = self.load_icon("./icons/load_icon.png", (24, 24))
        # New icons for extra features
        self.extract_audio_icon = self.load_icon("./icons/extract_audio.png", (24, 24))
        self.previous_icon = self.load_icon("./icons/previous.png", (24, 24))
        self.next_icon = self.load_icon("./icons/next.png", (24, 24))

        # Load icons for each loop mode.
        self.loop_icons = {
            "no_loop": self.load_icon("./icons/no_loop.png", (24, 24)),
            "loop": self.load_icon("./icons/loop.png", (24, 24)),
            "loop_current_file": self.load_icon("./icons/loop_current_file.png", (24, 24)),
            "shuffle": self.load_icon("./icons/shuffle.png", (24, 24))
        }

        # Volume icons
        self.volume_icons = {
            'mute': self.load_icon("./icons/volume_mute.png", (24, 24)),
            'low': self.load_icon("./icons/volume_low.png", (24, 24)),
            'medium': self.load_icon("./icons/volume_medium.png", (24, 24)),
            'high': self.load_icon("./icons/volume_high.png", (24, 24))
        }
        # Skip icons
        self.skip_forward_icon = self.load_icon("./icons/skip_5s_forward.png", (24, 24))
        self.skip_backward_icon = self.load_icon("./icons/skip_5s_backward.png", (24, 24))
        
        # Fast-Forward and Rewind icons
        self.fast_forward_icon = self.load_icon("./icons/fast_forward.png", (24, 24))
        self.fast_forward_active_icon = self.load_icon("./icons/fast_forward_active.png", (24, 24))
        self.rewind_icon = self.load_icon("./icons/rewind.png", (24, 24))
        self.rewind_active_icon = self.load_icon("./icons/rewind_active.png", (24, 24))
        
        # --------------------- Left Group Buttons ---------------------
        self.load_button = tk.Button(self.left_frame, image=self.load_icon_img, 
                                     command=self.load_video, bd=0, bg="gray")
        self.load_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.extract_audio_button = tk.Button(self.left_frame, image=self.extract_audio_icon,
                                              command=self.extract_audio, bd=0, bg="gray")
        self.extract_audio_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        # --------------------- Middle Group Buttons (Centered) ---------------------
        # Order: Previous, Skip Backward, Rewind, Play/Pause, Skip Forward, Fast-forward, Next
        self.previous_button = tk.Button(self.middle_frame, image=self.previous_icon,
                                         command=self.previous, bd=0, bg="gray")
        self.previous_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.rewind_button = tk.Button(self.middle_frame, image=self.rewind_icon,
                                       command=self.toggle_rewind, bd=0, bg="gray")
        self.rewind_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.skip_back_button = tk.Button(self.middle_frame, image=self.skip_backward_icon,
                                          command=lambda: self.skip_seconds(-5), bd=0, bg="gray")
        self.skip_back_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.toggle_button = tk.Button(self.middle_frame, image=self.play_icon,
                                       command=self.toggle_video, bd=0, bg="gray")
        self.toggle_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.skip_forward_button = tk.Button(self.middle_frame, image=self.skip_forward_icon,
                                             command=lambda: self.skip_seconds(5), bd=0, bg="gray")
        self.skip_forward_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.fast_forward_button = tk.Button(self.middle_frame, image=self.fast_forward_icon,
                                             command=self.toggle_fast_forward, bd=0, bg="gray")
        self.fast_forward_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.next_button = tk.Button(self.middle_frame, image=self.next_icon,
                                     command=self.next, bd=0, bg="gray")
        self.next_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        # --------------------- Right Group Buttons ---------------------
        self.loop_button = tk.Button(self.right_frame, image=self.loop_icons["no_loop"],
                                    command=self.toggle_loop, bd=0, bg="gray")
        self.loop_button.pack(side=tk.LEFT, padx=5, pady=1)
        
        # Volume controls remain on the far right
        self.volume_frame = tk.Frame(self.right_frame, bg="gray")
        self.volume_frame.pack(side=tk.LEFT, padx=5)
        
        self.mute_button = tk.Button(self.volume_frame, image=self.volume_icons['high'],
                                     command=self.toggle_mute, bd=0, bg="gray")
        self.mute_button.pack(side=tk.LEFT)
        
        self.volume_slider = tk.Scale(self.volume_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      bg="gray", fg="white", highlightthickness=0, length=100,
                                      sliderrelief=tk.FLAT)
        self.volume_slider.set(100)
        self.volume_slider.pack(side=tk.LEFT, padx=5)
        
        # Slider event bindings
        self.volume_slider.bind("<ButtonPress-1>", self.slider_pressed)
        self.volume_slider.bind("<ButtonRelease-1>", self.slider_released)
        self.volume_slider.bind("<B1-Motion>", self.slider_moved)
        
        # --------------------- Event Bindings ---------------------
        self.canvas.bind("<<SeekUpdate>>", self.handle_seek_update)
        self.canvas.bind("<<SeekComplete>>", self.handle_seek_complete)
        self.canvas.bind("<Button-1>", self.on_single_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.root.bind("f", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Configure>", self.debounced_resize)

        # Initialize updates
        self.update_progress()
        self.resize_debounce = None
        self.single_click_delay = None

    def load_icon(self, path, size=(24, 24)):
        """Load an icon with transparency support and resize it to keep buttons small."""
        from PIL import Image
        image = Image.open(path).convert("RGBA")
        image = image.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    # --------------------- New Feature Methods ---------------------
    def extract_audio(self):
        self.player.open_audio_extraction_window()

    def toggle_loop(self):
        current_mode = self.player.loop_mode
        modes = ["no_loop", "loop", "loop_current_file", "shuffle"]
        next_index = (modes.index(current_mode) + 1) % len(modes)
        new_mode = modes[next_index]
        self.player.loop_mode = new_mode
        print(f"Loop mode changed to: {new_mode}")
        self.loop_button.config(image=self.loop_icons[new_mode])

    # --------------------- Window Management ---------------------
    def debounced_resize(self, event):
        if self.resize_debounce:
            self.root.after_cancel(self.resize_debounce)
        self.resize_debounce = self.root.after(200, self.perform_resize)

    def perform_resize(self):
        if self.player.video_path:
            self.player.update_canvas_size(self.canvas.winfo_width(), self.canvas.winfo_height())
        self.resize_debounce = None

    # --------------------- Media Loading ---------------------
    def load_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[
                ("Media files", "*.mp4;*.avi;*.mkv;*.mp3;*.wav;*.m4a;*.ogg;*.flac"),
                ("All files", "*.*")
            ]
        )
        if video_path:
            self.player.load_video(video_path)
            self.player.play_video()
            self.toggle_button.config(image=self.pause_icon)
            self.perform_resize()

    def on_drop(self, event):
        """Handles file drag and drop"""
        video_path = event.data.strip()
        if video_path.startswith('{') and video_path.endswith('}'):
            video_path = video_path[1:-1]
        
        supported_extensions = (".mp4", ".avi", ".mkv", ".mp3", ".wav", ".m4a", ".ogg", ".flac")
        if video_path.lower().endswith(supported_extensions):
            self.player.load_video(video_path)
            self.player.play_video()
            self.toggle_button.config(image=self.pause_icon)
            self.perform_resize()
        else:
            print("Unsupported file type:", video_path)

    # --------------------- Playback Controls ---------------------
    def toggle_video(self):
        if self.player.paused:
            # If in no_loop mode and the file ended, restart from the beginning.
            if getattr(self.player, 'ended', False) and self.player.loop_mode == "no_loop":
                self.player.seek(0)
                self.player.ended = False
                self.player._end_handled = False
            self.player.play_video()
            self.toggle_button.config(image=self.pause_icon)
        else:
            self.player.pause_video()
            self.toggle_button.config(image=self.play_icon)

    def skip_seconds(self, seconds):
        """Immediate response with accumulated skips"""
        if not self.player.video_path:
            return
        
        # Calculate visual position first
        current_time = self.player.latest_frame_timestamp
        visual_time = current_time + seconds  # For immediate UI feedback
        visual_time = max(0, min(visual_time, self.player.total_time))
        
        # Update UI immediately
        self.progress_var.set((visual_time / self.player.total_time) * 100)
        self.update_progress_display(visual_time)
        
        # Queue actual seek
        self.player.seek_relative(seconds)
        
    # --------------------- Progress & Time Management ---------------------
    def update_progress(self):
        if self.player.video_path and self.player.total_time:
            progress = (self.player.latest_frame_timestamp / self.player.total_time) * 100
            self.progress_var.set(progress)
            cur_min, cur_sec = divmod(int(self.player.latest_frame_timestamp), 60)
            tot_min, tot_sec = divmod(int(self.player.total_time), 60)
            self.time_label.config(text=f"{cur_min:02d}:{cur_sec:02d} / {tot_min:02d}:{tot_sec:02d}")
        else:
            self.progress_var.set(0)
            self.time_label.config(text="00:00 / 00:00")
        self.root.after(200, self.update_progress)

    def update_progress_display(self, current_time):
        """Update time display without affecting actual playback"""
        cur_min, cur_sec = divmod(int(current_time), 60)
        tot_min, tot_sec = divmod(int(self.player.total_time), 60)
        self.time_label.config(text=f"{cur_min:02d}:{cur_sec:02d} / {tot_min:02d}:{tot_sec:02d}")

    # --------------------- Seeking Controls ---------------------
    def seek_video(self, event):
        """Seek to the clicked position in the progress bar."""
        if not self.player.video_path or not self.player.total_time:
            return
        new_time = (event.x / self.progress_bar.winfo_width()) * self.player.total_time
        self.player.seek(new_time)

    def scrub_video(self, event):
        """Lightweight scrub with visual-only updates"""
        if not self.player.video_path or not self.player.total_time:
            return
            
        new_time = (event.x / self.progress_bar.winfo_width()) * self.player.total_time
        self.progress_var.set((new_time / self.player.total_time) * 100)
        self.update_progress_display(new_time)

    def resume_video(self, event):
        """Final sync when releasing scrub bar"""
        new_time = (event.x / self.progress_bar.winfo_width()) * self.player.total_time
        self.player.seek(new_time)
        if self.player.paused:
            self.player.play_video()

    # --------------------- Volume Controls ---------------------
    def slider_pressed(self, event):
        self.slider_active = True

    def slider_released(self, event):
        self.slider_active = False
        self.handle_volume_change()

    def slider_moved(self, event):
        if self.slider_active:
            self.handle_volume_change()

    def handle_volume_change(self):
        new_volume = self.volume_slider.get() / 100
        
        if self.muted:
            # Unmute but keep original pre-mute volume
            self.muted = False
            self.volume = new_volume
        else:
            # Update both current and pre-mute volume
            self.volume = new_volume
            self.pre_mute_volume = new_volume
            
        self.player.set_volume(self.volume)
        self.update_volume_icon()

    def toggle_mute(self):
        """Toggle between muted and unmuted states"""
        if not self.muted:
            # Store current volume and mute
            self.pre_mute_volume = self.volume
            self.player.set_volume(0)
            self.volume_slider.set(0)
            self.muted = True
        else:
            # Restore original pre-mute volume
            self.player.set_volume(self.pre_mute_volume)
            self.volume_slider.set(self.pre_mute_volume * 100)
            self.volume = self.pre_mute_volume
            self.muted = False
            
        print(f"Mute State: {self.muted}")
        print(f"Current Volume: {self.volume}")
        print(f"Stored Pre-Mute Volume: {self.pre_mute_volume}")
        self.update_volume_icon()

    def update_volume_icon(self):
        """Update mute button icon based on volume level"""
        if self.volume <= 0.0 or self.muted:
            icon = 'mute'
        else:
            if self.pre_mute_volume <= 0.33:
                icon = 'low'
            elif self.pre_mute_volume <= 0.66:
                icon = 'medium'
            else:
                icon = 'high'
        self.mute_button.config(image=self.volume_icons[icon])

    # --------------------- Speed Controls ---------------------
    def toggle_fast_forward(self):
        current_speed = self.player.speed_multiplier
        new_speed = 4.0 if current_speed != 4.0 else 1.0
        self.player.set_speed(new_speed)
        self.update_speed_buttons()

    def toggle_rewind(self):
        current_speed = self.player.speed_multiplier
        new_speed = -4.0 if current_speed != -4.0 else 1.0
        self.player.set_speed(new_speed)
        self.update_speed_buttons()

    def update_speed_buttons(self):
        speed = self.player.speed_multiplier
        if speed == 4.0:
            self.fast_forward_button.config(image=self.fast_forward_active_icon)
            self.rewind_button.config(image=self.rewind_icon)
        elif speed == -4.0:
            self.fast_forward_button.config(image=self.fast_forward_icon)
            self.rewind_button.config(image=self.rewind_active_icon)
        else:
            self.fast_forward_button.config(image=self.fast_forward_icon)
            self.rewind_button.config(image=self.rewind_icon)

    # --------------------- Fullscreen Handling ---------------------
    def toggle_fullscreen(self, event=None):
        self.fullscreen = not getattr(self, 'fullscreen', False)
        self.root.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen(self, event=None):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)

    # --------------------- Event Handlers ---------------------
    def on_single_click(self, event):
        """Delay single click handling to avoid conflict with double-click."""
        if self.single_click_delay:
            self.root.after_cancel(self.single_click_delay)
        self.single_click_delay = self.root.after(300, self.toggle_video)

    def on_double_click(self, event):
        """Double-click toggles fullscreen and cancels single-click event."""
        if self.single_click_delay:
            self.root.after_cancel(self.single_click_delay)
            self.single_click_delay = None
        self.toggle_fullscreen()

    def handle_seek_complete(self, event):
        """Update UI after seek completes"""
        self.update_progress()
        self.canvas.update_idletasks()
        
    def handle_seek_update(self, event):
        self.player.update_display(force=True)

    def previous(self):
        """Go to the previous playable file in the directory."""
        self.player.load_previous_video()

    def next(self):
        """Go to the next playable file in the directory."""
        self.player.load_next_video()
