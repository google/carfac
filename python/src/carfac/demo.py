import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import threading
import queue
import time
from scipy.signal import butter, sosfilt, hilbert
from collections import deque
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Try to import carfac from multiple possible locations
pysai = None
try:
    # Method 1: Direct import (if installed)
    import carfac.sai as pysai
    print("✅ Using installed carfac.sai")
except ImportError:
    try:
        # Method 2: Add current directory structure to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            current_dir,  # Current directory
            os.path.dirname(current_dir),  # Parent directory
            os.path.join(current_dir, '..'),  # Up one level
            os.path.join(current_dir, '..', '..'),  # Up two levels
        ]
        
        # Add paths and try import
        for path in possible_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        import carfac.sai as pysai
        print("✅ Using local carfac.sai")
    except ImportError:
        print("⚠️ carfac.sai not found - using fallback pitch detection")
        pysai = None


class PitchVisualizer:
    def __init__(self, sample_rate=22050, chunk_size=512):
        # Audio settings
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Check if we have carfac
        self.use_carfac = pysai is not None
        
        if self.use_carfac:
            # SAI configuration
            self.sai_width = 100
            self.num_channels = 71
            self.input_segment_width = 882
            self._setup_sai()
            print("🎵 Using CARFAC-SAI pitch detection")
        else:
            # Simple autocorrelation parameters
            self.buffer_size = 4096
            print("🎵 Using autocorrelation pitch detection")
        
        # Display settings
        self.time_window = 8.0  # seconds
        self.pitch_range = (80, 800)  # Hz
        self.max_history = 1000
        
        # Linear wave effect settings
        self.wave_duration = 2.5  # How long each wave lasts
        self.max_wave_radius = 150  # Maximum wave radius in plot units
        self.wave_speed = 60  # Wave expansion speed
        self.num_wave_rings = 15  # Number of concentric linear rings
        
        # Data storage
        self.pitch_history = deque(maxlen=self.max_history)
        self.time_history = deque(maxlen=self.max_history)
        self.confidence_history = deque(maxlen=self.max_history)
        self.wave_sources = deque(maxlen=50)  # Store wave source points
        
        # Audio system
        self.audio_queue = queue.Queue(maxsize=50)
        self.is_running = False
        self.start_time = None
        
    def _setup_sai(self):
        """Initialize the SAI system"""
        if not self.use_carfac:
            return
            
        sai_params = pysai.SAIParams(
            num_channels=self.num_channels,
            input_segment_width=self.input_segment_width,
            trigger_window_width=self.input_segment_width,
            sai_width=self.sai_width,
            future_lags=self.sai_width // 2,
            num_triggers_per_frame=2
        )
        
        self.sai = pysai.SAI(sai_params)
        print(f"SAI initialized: {self.num_channels} channels, {self.sai_width} bins")
    
    def frequency_to_hsv_color(self, freq, min_freq=80, max_freq=800):
        """Convert frequency to HSV color with distinct color mapping per frequency range"""
        if freq <= 0 or freq < min_freq:
            return (0, 0, 0)  # Black for invalid frequencies
            
        # Clamp frequency to range
        freq = np.clip(freq, min_freq, max_freq)
        
        # Normalize frequency to 0-1
        normalized_freq = (freq - min_freq) / (max_freq - min_freq)
        
        # Create distinct HSV color zones
        # Each zone covers about 14% of the frequency range with distinct colors
        if normalized_freq <= 0.14:  # 80-180 Hz
            hue = 0.0  # Pure Red
            saturation = 0.9 + 0.1 * (normalized_freq / 0.14)
            value = 0.9 + 0.1 * (normalized_freq / 0.14)
        elif normalized_freq <= 0.28:  # 180-280 Hz
            hue = 0.08  # Orange-Red
            local_norm = (normalized_freq - 0.14) / 0.14
            saturation = 0.95
            value = 0.85 + 0.15 * local_norm
        elif normalized_freq <= 0.42:  # 280-380 Hz
            hue = 0.16  # Orange-Yellow
            local_norm = (normalized_freq - 0.28) / 0.14
            saturation = 1.0
            value = 0.9 + 0.1 * local_norm
        elif normalized_freq <= 0.56:  # 380-480 Hz
            hue = 0.25  # Yellow-Green
            local_norm = (normalized_freq - 0.42) / 0.14
            saturation = 0.9 + 0.1 * local_norm
            value = 0.95
        elif normalized_freq <= 0.7:  # 480-580 Hz
            hue = 0.35  # Green
            local_norm = (normalized_freq - 0.56) / 0.14
            saturation = 0.85 + 0.15 * local_norm
            value = 0.9
        elif normalized_freq <= 0.84:  # 580-680 Hz
            hue = 0.5  # Cyan-Blue
            local_norm = (normalized_freq - 0.7) / 0.14
            saturation = 0.9
            value = 0.85 + 0.15 * local_norm
        else:  # 680-800 Hz
            hue = 0.75  # Purple-Magenta
            local_norm = (normalized_freq - 0.84) / 0.16
            saturation = 0.95 - 0.1 * local_norm
            value = 0.8 + 0.2 * local_norm
        
        return (hue, saturation, value)
    
    def get_rainbow_colors(self, frequencies):
        """Get array of distinct HSV colors for given frequencies"""
        colors = []
        for freq in frequencies:
            hsv = self.frequency_to_hsv_color(freq)
            rgb = mcolors.hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
            colors.append(rgb)
        return colors
    
    def _autocorrelation_pitch(self, audio_segment):
        """Improved autocorrelation-based pitch detection"""
        # Apply window
        windowed = audio_segment * np.hanning(len(audio_segment))
        
        # Remove DC component
        windowed = windowed - np.mean(windowed)
        
        # Autocorrelation
        correlation = np.correlate(windowed, windowed, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Normalize
        if correlation[0] > 0:
            correlation = correlation / correlation[0]
        
        # Find peaks in autocorrelation
        min_period = int(self.sample_rate / 800)  # 800 Hz max
        max_period = int(self.sample_rate / 80)   # 80 Hz min
        
        if max_period >= len(correlation):
            return 0, 0
        
        # Look for peak in valid range
        search_range = correlation[min_period:max_period]
        if len(search_range) == 0:
            return 0, 0
            
        peak_idx = np.argmax(search_range)
        period = peak_idx + min_period
        
        # Calculate frequency and confidence
        frequency = self.sample_rate / period
        confidence = search_range[peak_idx]
        
        # Lower threshold for more detections
        if confidence < 0.15:
            return 0, 0
            
        return frequency, confidence
        
    def _create_filterbank(self, audio_segment):
        """Create multi-channel cochlear representation"""
        if len(audio_segment) != self.input_segment_width:
            audio_segment = np.resize(audio_segment, self.input_segment_width)
        
        channels = []
        freqs = np.logspace(np.log10(80), np.log10(8000), self.num_channels)
        
        for fc in freqs:
            try:
                # Simple bandpass filter
                bw = fc * 0.3
                low = max(fc - bw/2, 20)
                high = min(fc + bw/2, self.sample_rate/2 - 50)
                
                if low < high:
                    sos = butter(2, [low, high], btype='band', 
                               fs=self.sample_rate, output='sos')
                    filtered = sosfilt(sos, audio_segment)
                    envelope = np.abs(hilbert(filtered))
                    channels.append(envelope)
                else:
                    channels.append(np.zeros_like(audio_segment))
                    
            except Exception:
                channels.append(np.zeros_like(audio_segment))
        
        return np.array(channels, dtype=np.float32)
    
    def _extract_pitch(self, sai_frame):
        """Extract pitch from SAI frame"""
        if sai_frame is None or sai_frame.size == 0:
            return 0, 0
            
        # Average across channels
        summary = np.mean(sai_frame, axis=0) if len(sai_frame.shape) == 2 else sai_frame
        
        if len(summary) < 3:
            return 0, 0
            
        # Find peak
        max_idx = np.argmax(summary)
        max_val = summary[max_idx]
        
        if max_val < 0.0001:
            return 0, 0
            
        # Convert to frequency
        max_delay_ms = 12.5
        delay_ms = (max_idx / len(summary)) * max_delay_ms
        
        if delay_ms < 1.25:
            return 0, 0
            
        frequency = 1000.0 / delay_ms
        confidence = max_val / np.max(summary) if np.max(summary) > 0 else 0
        
        return frequency, confidence
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
            
        return (in_data, pyaudio.paContinue)
    
    def _processing_loop(self):
        """Main audio processing loop"""
        if self.use_carfac:
            audio_buffer = np.zeros(self.sample_rate * 2)
            buffer_pos = 0
        else:
            audio_buffer = np.zeros(4096)
            buffer_pos = 0
        
        while self.is_running:
            try:
                # Collect audio chunks
                chunks = []
                for _ in range(3):
                    try:
                        chunk = self.audio_queue.get_nowait()
                        chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if not chunks:
                    time.sleep(0.01)
                    continue
                    
                # Update buffer
                new_audio = np.concatenate(chunks)
                for sample in new_audio:
                    audio_buffer[buffer_pos] = sample
                    buffer_pos = (buffer_pos + 1) % len(audio_buffer)
                
                # Extract segment for processing
                if self.use_carfac:
                    start_idx = (buffer_pos - self.input_segment_width) % len(audio_buffer)
                    if start_idx + self.input_segment_width <= len(audio_buffer):
                        segment = audio_buffer[start_idx:start_idx + self.input_segment_width]
                    else:
                        part1 = audio_buffer[start_idx:]
                        part2 = audio_buffer[:self.input_segment_width - len(part1)]
                        segment = np.concatenate([part1, part2])
                else:
                    segment = audio_buffer.copy()
                
                # Normalize
                max_amp = np.max(np.abs(segment))
                if max_amp > 0.001:
                    segment = segment / max_amp
                    
                    if self.use_carfac:
                        cochlear_input = self._create_filterbank(segment)
                        sai_frame = self.sai.RunSegment(cochlear_input)
                        pitch, confidence = self._extract_pitch(sai_frame)
                    else:
                        pitch, confidence = self._autocorrelation_pitch(segment)
                    
                    # Store results and create wave source
                    current_time = time.time() - self.start_time
                    self.pitch_history.append(pitch)
                    self.confidence_history.append(confidence)
                    self.time_history.append(current_time)
                    
                    # Add wave source for significant detections
                    if pitch > 0 and confidence > 0.2:
                        self.wave_sources.append({
                            'time': current_time,
                            'pitch': pitch,
                            'confidence': confidence,
                            'x': current_time,
                            'y': pitch
                        })
                
                time.sleep(0.02)
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _create_linear_wave_rings(self, current_time):
        """Create expanding linear wave rings from pitch detections"""
        active_waves = []
        
        for source in list(self.wave_sources):
            age = current_time - source['time']
            
            if age > self.wave_duration:
                continue  # Wave has expired
            
            # Calculate wave expansion
            wave_radius = age * self.wave_speed
            
            if wave_radius > self.max_wave_radius:
                continue
            
            # Create multiple concentric LINEAR rings
            base_color = self.get_rainbow_colors([source['pitch']])[0]
            
            for ring_idx in range(self.num_wave_rings):
                ring_delay = ring_idx * 0.08  # Shorter delay for more rings
                ring_age = age - ring_delay
                
                if ring_age <= 0:
                    continue
                    
                ring_radius = ring_age * self.wave_speed
                
                if ring_radius > self.max_wave_radius:
                    continue
                
                # Fade out over time
                fade_factor = 1.0 - (ring_age / self.wave_duration)
                if fade_factor <= 0:
                    continue
                
                # Ring thickness and alpha
                ring_thickness = 2 + (source['confidence'] * 4)  # Thinner for linear look
                alpha = fade_factor * source['confidence'] * 0.7
                
                # Create linear wave ring data
                wave_data = {
                    'center_x': source['x'],
                    'center_y': source['y'],
                    'radius': ring_radius,
                    'thickness': ring_thickness,
                    'color': base_color,
                    'alpha': alpha,
                    'ring_index': ring_idx,
                    'age': ring_age
                }
                
                active_waves.append(wave_data)
        
        return active_waves
    
    def _draw_linear_waves(self, waves):
        """Draw linear wave rings (perfect circles)"""
        for wave in waves:
            center_x, center_y = wave['center_x'], wave['center_y']
            radius = wave['radius']
            color = wave['color']
            alpha = wave['alpha']
            thickness = wave['thickness']
            
            if alpha < 0.05 or radius < 1:
                continue
            
            # Create perfect circular wave using matplotlib Circle
            # But draw it as a thin line for linear appearance
            num_points = 128
            angles = np.linspace(0, 2*np.pi, num_points)
            
            # Calculate perfect circle points
            x_points = center_x + radius * np.cos(angles)
            y_points = center_y + radius * np.sin(angles)
            
            # Close the circle
            x_points = np.append(x_points, x_points[0])
            y_points = np.append(y_points, y_points[0])
            
            # Draw the linear wave ring as a thin perfect circle
            self.ax.plot(x_points, y_points, color=color, alpha=alpha, 
                        linewidth=thickness, solid_capstyle='round',
                        antialiased=True)
    
    def _freq_to_note(self, freq):
        """Convert frequency to musical note"""
        if freq <= 0:
            return "---"
        
        A4 = 440.0
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        semitones = 12 * np.log2(freq / A4)
        note_num = int(round(semitones)) + 9
        octave = 4 + (note_num // 12)
        note_idx = note_num % 12
        return f"{notes[note_idx]}{octave}"
    
    def _get_color_name(self, freq):
        """Get color name for frequency based on new HSV mapping"""
        normalized = (freq - self.pitch_range[0]) / (self.pitch_range[1] - self.pitch_range[0])
        
        if normalized <= 0.14:
            return "Red"
        elif normalized <= 0.28:
            return "Orange-Red" 
        elif normalized <= 0.42:
            return "Orange-Yellow"
        elif normalized <= 0.56:
            return "Yellow-Green"
        elif normalized <= 0.7:
            return "Green"
        elif normalized <= 0.84:
            return "Cyan-Blue"
        else:
            return "Purple-Magenta"
    
    def _update_plot(self, frame):
        """Update visualization with linear wave effects"""
        self.ax.clear()
        
        current_time = time.time() - self.start_time if self.start_time else 0
        
        if len(self.pitch_history) < 2:
            method = "CARFAC-SAI" if self.use_carfac else "Autocorrelation"
            self.ax.text(0.5, 0.5, f'📡 Listening for audio...\n({method} method)\n\nSpeak, sing, or play music!\nEach frequency gets its unique HSV color', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.7, color='cyan')
            self._setup_plot_appearance(current_time)
            return
        
        # Create and draw linear wave patterns
        waves = self._create_linear_wave_rings(current_time)
        self._draw_linear_waves(waves)
        
        # Add center points for active sources
        recent_sources = [s for s in self.wave_sources if (current_time - s['time']) < 0.3]
        if recent_sources:
            for source in recent_sources:
                age = current_time - source['time']
                alpha = 1.0 - (age / 0.3)
                color = self.get_rainbow_colors([source['pitch']])[0]
                
                self.ax.scatter(source['x'], source['y'], s=80, c=[color], 
                              alpha=alpha, edgecolors='white', linewidth=1.5, zorder=10)
        
        self._setup_plot_appearance(current_time)
        
        # Show current pitch info
        if len(self.pitch_history) > 0:
            recent_pitches = [(p, c) for p, c in zip(list(self.pitch_history)[-10:], 
                                                    list(self.confidence_history)[-10:]) 
                            if p > 0 and self.pitch_range[0] <= p <= self.pitch_range[1]]
            
            if recent_pitches:
                current_pitch, current_conf = recent_pitches[-1]
                note = self._freq_to_note(current_pitch)
                color_name = self._get_color_name(current_pitch)
                method = "CARFAC" if self.use_carfac else "Auto"
                
                title = f'📡 {current_pitch:.1f} Hz ({note}) - {color_name} [{method}] - Confidence: {current_conf:.2f}'
                self.ax.set_title(title, color='white', fontsize=12)
    
    def _setup_plot_appearance(self, current_time):
        """Setup plot appearance and limits"""
        # Set up plot limits
        self.ax.set_xlim(max(0, current_time - self.time_window), current_time + 1)
        self.ax.set_ylim(self.pitch_range[0], self.pitch_range[1])
        
        # Style
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Frequency (Hz)', color='white')
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.set_facecolor('black')
        
        # Show frequency color zones as reference
        zone_freqs = [80, 180, 280, 380, 480, 580, 680, 800]
        for i, freq in enumerate(zone_freqs[:-1]):
            if self.pitch_range[0] <= freq <= self.pitch_range[1]:
                color = self.get_rainbow_colors([freq + 50])[0]  # Mid-zone color
                self.ax.axhline(y=freq, color=color, alpha=0.15, linestyle='-', linewidth=0.8)
    
    def start(self):
        """Start the pitch visualizer"""
        method = "CARFAC-SAI" if self.use_carfac else "Autocorrelation"
        print(f"📡 Starting Linear Wave Pitch Visualizer ({method})...")
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self.start_time = time.time()
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Set up plot
            plt.style.use('dark_background')
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.fig.patch.set_facecolor('black')
            self.fig.suptitle(f'📡 Real-time Linear Wave Pitch Visualization - HSV Color Space ({method})', 
                            fontsize=16, color='white')
            
            # Start animation
            self.anim = FuncAnimation(self.fig, self._update_plot, 
                                    interval=50, blit=False)
            
            print("🎵 Visualizer started! Speak, sing, or play audio...")
            print("📡 Watch perfect circular waves expand from each pitch!")
            print("🌈 HSV Color Zones:")
            print("   80-180 Hz  → Red")
            print("   180-280 Hz → Orange-Red") 
            print("   280-380 Hz → Orange-Yellow")
            print("   380-480 Hz → Yellow-Green")
            print("   480-580 Hz → Green")
            print("   580-680 Hz → Cyan-Blue")
            print("   680-800 Hz → Purple-Magenta")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error starting audio: {e}")
            self.stop()
    
    def stop(self):
        """Stop the visualizer"""
        print("🛑 Stopping...")
        self.is_running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    """Main function"""
    print("=" * 60)
    print("📡 CARFAC/Linear Wave Pitch Visualizer 📡")
    print("=" * 60)
    
    try:
        visualizer = PitchVisualizer()
        visualizer.start()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()