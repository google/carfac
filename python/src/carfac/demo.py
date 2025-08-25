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
        self.time_window = 10.0  # seconds
        self.pitch_range = (50, 400)  # Hz
        self.max_history = 1000
        
        # Data storage
        self.pitch_history = deque(maxlen=self.max_history)
        self.time_history = deque(maxlen=self.max_history)
        self.confidence_history = deque(maxlen=self.max_history)
        
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
    
    def frequency_to_hsv_color(self, freq, min_freq=50, max_freq=400):
        """Convert frequency to HSV color for natural rainbow effect"""
        if freq <= 0 or freq < min_freq:
            return (0, 0, 0)  # Black for invalid frequencies
            
        # Clamp frequency to range
        freq = np.clip(freq, min_freq, max_freq)
        
        # Normalize frequency to 0-1
        normalized_freq = (freq - min_freq) / (max_freq - min_freq)
        
        # Map to hue - Rainbow progression: Red -> Orange -> Yellow -> Green -> Blue -> Purple
        # Low freq = red (hue=0), High freq = purple (hue=0.83)
        hue = (1 - normalized_freq) * 0.83
        
        saturation = 1.0  # Full saturation for vibrant colors
        value = 0.9      # Bright but not maximum
        
        return (hue, saturation, value)
    
    def get_rainbow_colors(self, frequencies):
        """Get array of rainbow colors for given frequencies"""
        colors = []
        for freq in frequencies:
            hsv = self.frequency_to_hsv_color(freq)
            rgb = mcolors.hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
            colors.append(rgb)
        return colors
    
    def _autocorrelation_pitch(self, audio_segment):
        """Simple autocorrelation-based pitch detection (fallback method)"""
        # Apply window
        windowed = audio_segment * np.hanning(len(audio_segment))
        
        # Autocorrelation
        correlation = np.correlate(windowed, windowed, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Find peaks in autocorrelation
        min_period = int(self.sample_rate / 400)  # 400 Hz max
        max_period = int(self.sample_rate / 50)   # 50 Hz min
        
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
        confidence = search_range[peak_idx] / np.max(correlation) if np.max(correlation) > 0 else 0
        
        # Threshold for valid detection
        if confidence < 0.3:
            return 0, 0
            
        return frequency, confidence
        
    def _create_filterbank(self, audio_segment):
        """Create multi-channel cochlear representation"""
        if len(audio_segment) != self.input_segment_width:
            audio_segment = np.resize(audio_segment, self.input_segment_width)
        
        channels = []
        freqs = np.logspace(np.log10(80), np.log10(4000), self.num_channels)
        
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
        
        if max_val < 0.001:
            return 0, 0
            
        # Convert to frequency
        max_delay_ms = 25.0
        delay_ms = (max_idx / len(summary)) * max_delay_ms
        
        if delay_ms < 0.5:
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
            audio_buffer = np.zeros(self.sample_rate * 2)  # 2 second buffer
            buffer_pos = 0
        else:
            audio_buffer = np.zeros(4096)
            buffer_pos = 0
        
        while self.is_running:
            try:
                # Collect audio chunks
                chunks = []
                for _ in range(3):  # Process multiple chunks together
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
                        # Handle wrap-around
                        part1 = audio_buffer[start_idx:]
                        part2 = audio_buffer[:self.input_segment_width - len(part1)]
                        segment = np.concatenate([part1, part2])
                else:
                    # Use entire buffer for autocorrelation
                    segment = audio_buffer.copy()
                
                # Normalize
                max_amp = np.max(np.abs(segment))
                if max_amp > 0.001:  # Only process if there's significant audio
                    segment = segment / max_amp
                    
                    if self.use_carfac:
                        # Create filterbank
                        cochlear_input = self._create_filterbank(segment)
                        
                        # Run SAI
                        sai_frame = self.sai.RunSegment(cochlear_input)
                        
                        # Extract pitch
                        pitch, confidence = self._extract_pitch(sai_frame)
                    else:
                        # Use autocorrelation
                        pitch, confidence = self._autocorrelation_pitch(segment)
                    
                    # Store results
                    current_time = time.time() - self.start_time
                    self.pitch_history.append(pitch)
                    self.confidence_history.append(confidence)
                    self.time_history.append(current_time)
                
                time.sleep(0.02)  # 50 Hz processing
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
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
        """Get color name for frequency"""
        normalized = (freq - self.pitch_range[0]) / (self.pitch_range[1] - self.pitch_range[0])
        
        if normalized <= 0.17:
            return "Purple"
        elif normalized <= 0.33:
            return "Blue"
        elif normalized <= 0.5:
            return "Green"
        elif normalized <= 0.67:
            return "Yellow"
        elif normalized <= 0.83:
            return "Orange"
        else:
            return "Red"
    
    def _update_plot(self, frame):
        """Update visualization with HSV rainbow colors"""
        self.ax.clear()
        
        if len(self.pitch_history) < 2:
            method = "CARFAC-SAI" if self.use_carfac else "Autocorrelation"
            self.ax.text(0.5, 0.5, f'🎤 Listening for audio...\n({method} method)', 
                        transform=self.ax.transAxes, ha='center', va='center',
                        fontsize=14, alpha=0.7)
            return
        
        # Get data
        times = np.array(self.time_history)
        pitches = np.array(self.pitch_history)
        confidences = np.array(self.confidence_history)
        
        # Filter valid pitches in display range
        valid_mask = (pitches > 0) & (pitches >= self.pitch_range[0]) & (pitches <= self.pitch_range[1])
        
        if np.any(valid_mask):
            valid_times = times[valid_mask]
            valid_pitches = pitches[valid_mask]
            valid_confidences = confidences[valid_mask]
            
            # Get rainbow colors based on frequency
            colors = self.get_rainbow_colors(valid_pitches)
            
            # Create scatter plot with HSV rainbow coloring
            # Size based on confidence
            sizes = 20 + (valid_confidences * 40)  # 20-60 point size
            
            scatter = self.ax.scatter(valid_times, valid_pitches, 
                                    c=colors,
                                    s=sizes, 
                                    alpha=0.8,
                                    edgecolors='white',
                                    linewidth=0.5)
        
        # Add frequency color bar reference
        self._add_frequency_colorbar()
        
        # Set up plot
        current_time = time.time() - self.start_time if self.start_time else 0
        self.ax.set_xlim(max(0, current_time - self.time_window), current_time + 1)
        self.ax.set_ylim(self.pitch_range[0], self.pitch_range[1])
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.grid(True, alpha=0.3)
        
        # Show current pitch with its rainbow color
        if len(pitches) > 0:
            current_pitch = pitches[-1]
            current_conf = confidences[-1]
            
            if current_pitch > 0 and self.pitch_range[0] <= current_pitch <= self.pitch_range[1]:
                note = self._freq_to_note(current_pitch)
                color_name = self._get_color_name(current_pitch)
                method = "CARFAC" if self.use_carfac else "Auto"
                
                title = f'🌈 {current_pitch:.1f} Hz ({note}) - {color_name} [{method}] - Confidence: {current_conf:.2f}'
                self.ax.set_title(title)
    
    def _add_frequency_colorbar(self):
        """Add frequency-to-color reference lines"""
        key_freqs = [75, 125, 175, 225, 275, 325, 375]
        for freq in key_freqs:
            if self.pitch_range[0] <= freq <= self.pitch_range[1]:
                color = self.get_rainbow_colors([freq])[0]
                self.ax.axhline(y=freq, color=color, alpha=0.3, linestyle='--', linewidth=1)
    
    def start(self):
        """Start the pitch visualizer"""
        method = "CARFAC-SAI" if self.use_carfac else "Autocorrelation"
        print(f"🌈 Starting Rainbow Pitch Visualizer ({method})...")
        
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
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            self.fig.suptitle(f'🌈 Real-time Rainbow Pitch Detection ({method})', fontsize=14)
            
            # Start animation
            self.anim = FuncAnimation(self.fig, self._update_plot, 
                                    interval=50, blit=False)
            
            print("🎵 Visualizer started! Speak, sing, or play audio...")
            print("🌈 Colors: Red (low) → Orange → Yellow → Green → Blue → Purple (high)")
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
    print("🌈 CARFAC/Rainbow Pitch Visualizer 🌈")
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