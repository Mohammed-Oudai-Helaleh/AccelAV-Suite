import tkinter as tk

class TimeInputField(tk.Frame):
    def __init__(self, master, initial_time="00:00:00:000", **kwargs):
        """
        A custom time input widget in HH:MM:SS:ms format.
        """
        super().__init__(master, **kwargs)
        try:
            hours, minutes, seconds, millis = initial_time.split(":")
        except ValueError:
            hours, minutes, seconds, millis = "00", "00", "00", "000"
        
        # StringVars for each time component.
        self.hour_var = tk.StringVar(value=hours.zfill(2))
        self.minute_var = tk.StringVar(value=minutes.zfill(2))
        self.second_var = tk.StringVar(value=seconds.zfill(2))
        self.millisecond_var = tk.StringVar(value=millis.zfill(3))
        
        # Register validation functions.
        vcmd_hour = (self.register(self.validate_hour), '%P')
        vcmd_min = (self.register(self.validate_minute), '%P')
        vcmd_sec = (self.register(self.validate_second), '%P')
        vcmd_ms = (self.register(self.validate_millisecond), '%P')
        
        # Build widget layout with colon separators.
        self.hour_entry = tk.Entry(self, width=2, textvariable=self.hour_var,
                                   validate='key', validatecommand=vcmd_hour)
        self.hour_entry.pack(side="left")
        
        tk.Label(self, text=":").pack(side="left")
        self.minute_entry = tk.Entry(self, width=2, textvariable=self.minute_var,
                                     validate='key', validatecommand=vcmd_min)
        self.minute_entry.pack(side="left")
        
        tk.Label(self, text=":").pack(side="left")
        self.second_entry = tk.Entry(self, width=2, textvariable=self.second_var,
                                     validate='key', validatecommand=vcmd_sec)
        self.second_entry.pack(side="left")
        
        tk.Label(self, text=":").pack(side="left")
        self.millisecond_entry = tk.Entry(self, width=3, textvariable=self.millisecond_var,
                                          validate='key', validatecommand=vcmd_ms)
        self.millisecond_entry.pack(side="left")
    
    def validate_hour(self, value):
        if value == "":
            return True
        if value.isdigit():
            return True
        return False
    
    def validate_minute(self, value):
        if value == "":
            return True
        if value.isdigit() and 0 <= int(value) < 60:
            return True
        return False

    def validate_second(self, value):
        if value == "":
            return True
        if value.isdigit() and 0 <= int(value) < 60:
            return True
        return False

    def validate_millisecond(self, value):
        if value == "":
            return True
        if value.isdigit() and 0 <= int(value) < 1000:
            return True
        return False

    def get_time_in_seconds(self):
        """
        Convert the current input into total seconds (float).
        """
        try:
            hours = int(self.hour_var.get())
            minutes = int(self.minute_var.get())
            seconds = int(self.second_var.get())
            milliseconds = int(self.millisecond_var.get())
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        except ValueError:
            raise ValueError("Invalid time input in one or more fields.")

    def get_time_str(self):
        """
        Return the time in HH:MM:SS:ms format with proper zero-padding.
        """
        try:
            hours = int(self.hour_var.get())
            minutes = int(self.minute_var.get())
            seconds = int(self.second_var.get())
            milliseconds = int(self.millisecond_var.get())
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
        except ValueError:
            return "00:00:00:000"