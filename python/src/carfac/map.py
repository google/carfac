#!/usr/bin/env python3
"""
Real-time Stabilized Auditory Image (SAI) animation using microphone input.
Requires: pyaudio, numpy, matplotlib, and the CARFAC library.

This script captures audio from the microphone, processes it through CARFAC
to get neural activity patterns, then computes and displays SAI frames
in real-time as an animated heatmap.
"""

import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import threading
import queue
import time
from typing import Optional

# Assuming you have the CARFAC library available
# If using the JAX version from the first document:
# import carfac_jax as carfac
# For the Python version with SAI:
# import carfac.carfac as carfac
# import carfac.sai as sai

# For this example, I'll create simplified placeholder classes
# Replace these with actual CARFAC imports

class SimplifiedCARFAC:
    """Simplified CARFAC placeholder - replace with actual CARFAC implementation."""
    
    def __init__(self, fs=22050, n_ch=71):
        self.fs = fs
        self.n_ch = n_ch
        # Initialize your actual CARFAC model here
        
    def process(self, audio_segment):
        """Process audio through CARFAC filterbank."""
        # This is a placeholder - replace with actual CARFAC processing
        # Should return neural activity pattern (NAP) with shape (n_ch, len(audio_segment))
        
        # Simulate cochlear filtering with simple filterbank
        frequencies = np.logspace(np.log10(80), np.log10(8000), self.n_ch)
        nap = np.zeros((self.n_ch, len(audio_segment)))
        
        for i, freq in enumerate(frequencies):
            # Simple bandpass filter simulation
            dt = 1.0 / self.fs
            t = np.arange(len(audio_segment)) * dt
            
            # Create simple resonant filter response
            omega = 2 * np.pi * freq
            decay = np.exp(-omega * 0.01 * t)
            response = np.convolve(audio_segment, decay[:100], mode='same')
            
            # Half-wave rectification and scaling
            nap[i, :] = np.maximum(0, response) * (1 + 0.1 * np.random.randn(len(response)))
            
        return nap

class SimplifiedSAI:
    """Simplified SAI implementation based on the provided code."""
    
    def __init__(self, num_channels=71, sai_width=100, trigger_window_width=200, chunk_size=1024):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.trigger_window_width = trigger_window_width
        self.future_lags = sai_width // 4  # Quarter from future
        self.num_triggers_per_frame = 3
        
        # Buffer for accumulating input - ensure it's large enough for chunk_size
        min_buffer_width = sai_width + int(
            (1 + float(self.num_triggers_per_frame - 1) / 2) * trigger_window_width
        )
        self.buffer_width = max(min_buffer_width, chunk_size * 2)  # At least 2x chunk size
        self.input_buffer = np.zeros((num_channels, self.buffer_width))
        
        # Window function for trigger detection
        self.window = np.sin(np.linspace(
            np.pi / trigger_window_width, np.pi, trigger_window_width
        ))
        
    def process_segment(self, nap_segment):
        """Process NAP segment to generate SAI frame."""
        # Handle case where input segment is larger than buffer capacity
        segment_width = nap_segment.shape[1]
        
        if segment_width >= self.buffer_width:
            # If input is larger than buffer, just use the most recent part
            self.input_buffer = nap_segment[:, -self.buffer_width:]
        else:
            # Normal case: shift buffer and add new input
            overlap_width = self.buffer_width - segment_width
            if overlap_width > 0:
                self.input_buffer[:, :overlap_width] = self.input_buffer[:, -overlap_width:]
                self.input_buffer[:, overlap_width:] = nap_segment
            else:
                # Edge case: exact fit
                self.input_buffer = nap_segment
        
        # Generate SAI frame
        sai_frame = self._stabilize_segment()
        return sai_frame
        
    def _stabilize_segment(self):
        """Generate SAI frame using trigger-based correlation."""
        output_buffer = np.zeros((self.num_channels, self.sai_width))
        
        num_samples = self.input_buffer.shape[1]
        window_hop = self.trigger_window_width // 2
        window_start = (num_samples - self.trigger_window_width) - \
                      (self.num_triggers_per_frame - 1) * window_hop
        
        window_range_start = window_start - self.future_lags
        offset_range_start = 1 + window_start - self.sai_width
        
        if offset_range_start <= 0:
            return output_buffer
            
        for i in range(self.num_channels):
            nap_wave = self.input_buffer[i, :]
            
            for w in range(self.num_triggers_per_frame):
                current_window_offset = w * window_hop
                current_window_start = window_range_start + current_window_offset
                
                if current_window_start < 0 or current_window_start + self.trigger_window_width >= num_samples:
                    continue
                    
                # Find trigger point
                trigger_window = nap_wave[
                    current_window_start:current_window_start + self.trigger_window_width
                ]
                
                if len(trigger_window) == len(self.window):
                    windowed_signal = trigger_window * self.window
                    peak_val = np.max(windowed_signal)
                    trigger_time = np.argmax(windowed_signal) + current_window_offset
                    
                    if peak_val <= 0:
                        peak_val = np.max(self.window)
                        trigger_time = np.argmax(self.window) + current_window_offset
                    
                    # Blend segment into output
                    alpha = (0.025 + peak_val) / (0.5 + peak_val)
                    
                    start_idx = trigger_time + offset_range_start
                    end_idx = start_idx + self.sai_width
                    
                    if start_idx >= 0 and end_idx <= num_samples:
                        output_buffer[i, :] *= (1 - alpha)
                        output_buffer[i, :] += alpha * nap_wave[start_idx:end_idx]
        
        return output_buffer

class RealTimeSAIAnimator:
    """Real-time SAI animator with microphone input."""
    
    def __init__(self, 
                 chunk_size=1024, 
                 sample_rate=22050, 
                 n_channels=71,
                 sai_width=100,
                 update_interval=50):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.sai_width = sai_width
        self.update_interval = update_interval
        
        # Audio setup
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # CARFAC and SAI setup
        self.carfac = SimplifiedCARFAC(fs=sample_rate, n_ch=n_channels)
        self.sai = SimplifiedSAI(
            num_channels=n_channels, 
            sai_width=sai_width,
            trigger_window_width=min(400, chunk_size),
            chunk_size=chunk_size
        )
        
        # Visualization setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.sai_data = np.zeros((n_channels, sai_width))
        
        # Create rainbow HSV colormap
        cmap = self.create_rainbow_colormap()
        
    def create_rainbow_colormap(self):
        """Create rainbow HSV colormap: Blue→Cyan→Green→Yellow→Orange→Red."""
        from matplotlib.colors import hsv_to_rgb, ListedColormap
        n_bins = 256
        
        # Rainbow: Full spectrum from blue to red
        hues = np.linspace(0.7, 0.0, n_bins)     # Blue to red
        saturations = np.ones(n_bins) * 0.9       # High saturation throughout
        values = np.linspace(0.2, 1.0, n_bins)   # Dark to bright
        
        hsv_colors = np.stack([hues, saturations, values], axis=-1)
        rgb_colors = hsv_to_rgb(hsv_colors.reshape(-1, 1, 3)).reshape(-1, 3)
        
        return ListedColormap(rgb_colors)
        
        self.im = self.ax.imshow(
            self.sai_data, 
            aspect='auto', 
            origin='lower',
            cmap=cmap,
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        
        self.ax.set_title('Real-time Stabilized Auditory Image (SAI)', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time Lag (samples)', fontsize=12)
        self.ax.set_ylabel('Frequency Channel', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8)
        cbar.set_label('Correlation Strength', fontsize=10)
        
        # Add frequency labels (approximate)
        freq_labels = np.logspace(np.log10(80), np.log10(8000), 8).astype(int)
        freq_positions = np.linspace(0, n_channels-1, len(freq_labels))
        self.ax.set_yticks(freq_positions)
        self.ax.set_yticklabels([f'{f}Hz' for f in freq_labels])
        
        # Add lag labels
        lag_ms = np.arange(0, sai_width, sai_width//5) * 1000 / sample_rate
        lag_positions = np.arange(0, sai_width, sai_width//5)
        self.ax.set_xticks(lag_positions)
        self.ax.set_xticklabels([f'{lag:.1f}ms' for lag in lag_ms])
        
        plt.tight_layout()
        
        # Audio stream initialization
        self.p = None
        self.stream = None
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")
            
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # Skip if queue is full (processing can't keep up)
            pass
            
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Audio processing thread."""
        while self.running:
            try:
                # Get audio data with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process through CARFAC
                nap = self.carfac.process(audio_chunk)
                
                # Process through SAI
                sai_frame = self.sai.process_segment(nap)
                
                # Update visualization data
                self.sai_data = sai_frame
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
    
    def update_plot(self, frame):
        """Update the plot for animation."""
        if self.sai_data is not None:
            # Apply some smoothing and normalization
            smoothed_data = self.sai_data.copy()
            
            # Normalize to [0, 1] range with some dynamic range compression
            max_val = np.percentile(smoothed_data, 95)
            if max_val > 0:
                smoothed_data = np.clip(smoothed_data / max_val, 0, 1)
            
            # Apply gamma correction for better visualization
            smoothed_data = np.power(smoothed_data, 0.7)
            
            self.im.set_data(smoothed_data)
            
            # Update title with current time
            self.ax.set_title(
                f'Real-time Stabilized Auditory Image (SAI) - {time.strftime("%H:%M:%S")}',
                fontsize=14, fontweight='bold'
            )
        
        return [self.im]
    
    def start(self):
        """Start the real-time SAI animation."""
        print("Starting real-time SAI animation...")
        print("Speak or play music near the microphone to see the SAI patterns!")
        
        try:
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Start audio stream
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
            
            self.running = True
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self.process_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            # Start animation
            self.ani = animation.FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=self.update_interval,
                blit=True,
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            print(f"Error starting animation: {e}")
            self.stop()
    
    def stop(self):
        """Stop the animation and clean up."""
        print("Stopping SAI animation...")
        
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
        
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        
        plt.close('all')

def main():
    """Main function to run the real-time SAI animation."""
    print("Real-time SAI Animation with Rainbow Color Mapping")
    print("=================================================")
    print("This will capture audio from your microphone and display")
    print("a real-time Stabilized Auditory Image (SAI) animation.")
    print()
    print("Color mapping: Rainbow (Blue→Cyan→Green→Yellow→Orange→Red)")
    print("- Low correlation: Dark blue")
    print("- High correlation: Bright red")
    print()
    print("Requirements:")
    print("- pyaudio: pip install pyaudio")
    print("- numpy, matplotlib: pip install numpy matplotlib")
    print("- Working microphone")
    print()
    
    try:
        # Create and start the animator
        animator = RealTimeSAIAnimator(
            chunk_size=1024,
            sample_rate=22050,
            n_channels=71,
            sai_width=150,
            update_interval=50  # 20 FPS
        )
        
        animator.start()
        
    except KeyboardInterrupt:
        print("\nStopping animation...")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. A working microphone")
        print("2. Required packages installed: pip install pyaudio numpy matplotlib")
        print("3. Proper audio permissions")
    
    finally:
        if 'animator' in locals():
            animator.stop()

if __name__ == "__main__":
    main()