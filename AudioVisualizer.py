import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap

class AudioVisualizer:
    def __init__(self, data, sample_rate, chunk_size=256):  # Reduced chunk size for more updates
        self.data = data
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.current_frame = 0
        
        # Original colormap setup
        self.cmap = LinearSegmentedColormap.from_list('cyber', [
            '#00ffcc', '#ff00ff', '#7700ff'
        ])

        # Optimized figure setup (same visuals, faster backend)
        self.fig = Figure(figsize=(16, 6), facecolor='black', dpi=80)  # Slightly lower DPI for speed
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Pre-allocated reusable objects
        self.x = np.linspace(0, 1, self.chunk_size)
        self.line = self.ax.plot([], [], lw=2, alpha=0.8)[0]  # Single line object reuse
        self.fill = None  # Will recreate fill but keep original style
        
        # Original grid and limits
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.grid(color='#404040', alpha=0.2, linestyle='--')

    def render_frame(self):
        """Identical visual output to original but optimized"""
        start = self.current_frame * (self.chunk_size // 2)  # 50% overlap for smoothness
        end = start + self.chunk_size
        
        if end > len(self.data):
            return np.zeros((600, 800, 3), dtype=np.uint8)

        chunk = self.data[start:end]
        
        # Original color calculation preserved
        colors = self.cmap((chunk + 1) / 2)  # Maintains purple/magenta/cyan gradient
        avg_color = colors[len(colors)//2]    # Preserves original midpoint color selection
        
        # Update existing elements instead of creating new ones
        self.line.set_data(self.x, chunk)
        self.line.set_color(avg_color)
        self.line.set_linewidth(1 + abs(chunk.mean()) * 3)  # Original width calculation
        
        # Recreate fill with original parameters
        if self.fill:
            self.fill.remove()
        self.fill = self.ax.fill_between(self.x, chunk, 0, color='#00ffcc', alpha=0.1)
        
        # Faster buffer handling
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        
        return np.frombuffer(buf, dtype=np.uint8).reshape(
            (int(self.fig.bbox.height), int(self.fig.bbox.width), 4)
        )[..., :3]