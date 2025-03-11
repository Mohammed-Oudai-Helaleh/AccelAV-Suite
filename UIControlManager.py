import tkinter as tk
from typing import Optional

class UIControlManager:
    def __init__(self, canvas: tk.Canvas, control_frame: tk.Frame):
        """
        Initialize UI control manager
        :param canvas: Reference to the main video canvas
        :param control_frame: Reference to the controls frame
        """
        self.canvas = canvas
        self.control_frame = control_frame
        
        # Configuration parameters
        self.hide_controls_after = 5000  # ms before hiding controls
        self.sensitive_area_height = 60  # Bottom area for control activation
        self.cursor_hide_delay = 3000     # ms before hiding cursor
        
        # State tracking
        self.mouse_visible: bool = True
        self.hide_after_id: Optional[str] = None  # Control hide timer ID
        self.mouse_hide_timer: Optional[str] = None  # Cursor hide timer ID

        # Setup event bindings
        self._setup_event_handlers()

    # --------------------- Event Handler Setup ---------------------
    def _setup_event_handlers(self):
        """Configure all canvas and control frame event bindings"""
        self.canvas.bind("<Motion>", self.handle_canvas_motion)
        self.canvas.bind("<Enter>", self.handle_canvas_enter)
        self.canvas.bind("<Leave>", self.handle_canvas_leave)
        
        self.control_frame.bind("<Enter>", self.handle_control_enter)
        self.control_frame.bind("<Leave>", self.handle_control_leave)

    # --------------------- Control Visibility Methods ---------------------
    def show_controls(self, event=None):
        """Display controls with proper positioning"""
        self.control_frame.place(
            in_=self.canvas,
            relx=0.5, 
            rely=1.0, 
            anchor="s",
            relwidth=1.0,
            y=-2
        )
        self.control_frame.lift()
        self.reset_hide_timer()
        
    def hide_controls(self):
        """Hide controls if mouse not in sensitive area"""
        if not self.is_mouse_in_sensitive_area(self.canvas.winfo_pointery() - self.canvas.winfo_rooty()):
            self.control_frame.place_forget()
        self.hide_after_id = None
        
    def reset_hide_timer(self):
        """Reset the auto-hide countdown for controls"""
        self.cancel_hide()
        self.hide_after_id = self.canvas.after(self.hide_controls_after, self.hide_controls)
        
    def cancel_hide(self):
        """Cancel pending control hide operation"""
        if self.hide_after_id:
            self.canvas.after_cancel(self.hide_after_id)
            self.hide_after_id = None
            
    def schedule_hide(self):
        """Schedule controls to hide after timeout"""
        if not self.hide_after_id:
            self.hide_after_id = self.canvas.after(
                self.hide_controls_after, 
                self.hide_controls
            )
            
    # --------------------- Mouse Interaction Methods ---------------------
    def is_mouse_in_sensitive_area(self, y_pos) -> bool:
        """Check if cursor is in bottom control-sensitive area"""
        canvas_height = self.canvas.winfo_height()
        return y_pos > (canvas_height - self.sensitive_area_height)
    
    def handle_canvas_motion(self, event: tk.Event):
        """Handle mouse movement over canvas"""
        self.show_mouse()
        if self.is_mouse_in_sensitive_area(event.y):
            self.show_controls()
            self.cancel_hide()
        else:
            self.schedule_hide()
        self.reset_mouse_hide_timer()
        
    def handle_canvas_enter(self, event: tk.Event):
        """Handle mouse entering canvas area"""

    def handle_canvas_leave(self, event: tk.Event):
        """Handle mouse leaving canvas area"""
        self.show_mouse()
        self.schedule_hide()
        
    # --------------------- Cursor Management ---------------------
    def reset_mouse_hide_timer(self):
        """Reset countdown for cursor auto-hide"""
        if self.mouse_hide_timer:
            self.canvas.after_cancel(self.mouse_hide_timer)
        self.mouse_hide_timer = self.canvas.after(self.cursor_hide_delay, self.hide_mouse)
        
    def hide_mouse(self):
        """Hide canvas cursor"""
        y_pos = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        if not self.is_mouse_in_sensitive_area(y_pos):
            self.canvas.config(cursor='none')
            self.mouse_visible = False
            
    def show_mouse(self, event=None):
        """Show canvas cursor and reset timer"""
        if not self.mouse_visible:
            self.canvas.config(cursor='')
            self.mouse_visible = True
        self.reset_mouse_hide_timer()
        
    # --------------------- Control Frame Events ---------------------
    def handle_control_enter(self, event: tk.Event):
        """Handle mouse entering control frame"""
        self.cancel_hide()
        
    def handle_control_leave(self, event: tk.Event):
        """Handle mouse leaving control frame"""
        self.schedule_hide()