#!/usr/bin/env python3
"""
Real-time SAI animation using actual CARFAC-JAX implementation.
This uses the full research-grade cochlear model from the JAX folder.
"""

import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb, ListedColormap
import threading
import queue
import time
import sys
import os

# Add the JAX folder to Python path to import CARFAC
sys.path.append('./jax')  # Adjust path as needed

try:
    import jax
    import jax.numpy as jnp
    import carfac  # Import the actual CARFAC-JAX implementation
    print("Successfully imported CARFAC-JAX!")
except ImportError as e:
    print(f"Error importing CARFAC-JAX: {e}")
    print("Make sure the JAX folder is in the correct path and JAX is installed.")
    sys.exit(1)

class RealCARFACProcessor:
    """Wrapper for the actual CARFAC-JAX implementation."""
    
    def __init__(self, fs=22050, n_channels=71):
        self.fs = fs
        
        print("Initializing CARFAC-JAX model...")
        
        # Create CARFAC design parameters
        self.params = carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        
        # Design and initialize the CARFAC model
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(self.params)
        
        # Get actual number of channels from the model
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.pole_freqs = self.hypers.ears[0].pole_freqs
        
        print(f"CARFAC initialized with {self.n_channels} channels")
        print(f"Frequency range: {float(self.pole_freqs[-1]):.1f}Hz - {float(self.pole_freqs[0]):.1f}Hz")
        
        # JIT compile the processing function for speed
        print("JIT-compiling CARFAC processing...")
        self.run_segment_jit = jax.jit(
            carfac.run_segment,
            static_argnames=['hypers', 'open_loop']
        )
        
    def process_chunk(self, audio_chunk):
        """Process audio chunk through CARFAC-JAX."""
        # Reshape audio for CARFAC (needs to be 2D: [time, ears])
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)  # [time, 1_ear]
        else:
            audio_input = audio_chunk
            
        # Convert to JAX array
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        
        # Process through CARFAC
        naps, _, self.state, bm_output, _, _ = self.run_segment_jit(
            audio_jax, self.hypers, self.weights, self.state, open_loop=False
        )
        
        # Extract NAP for ear 0, transpose to [channels, time]
        nap_output = np.array(naps[:, 0, :])  # [time, channels] -> [channels, time]
        
        return nap_output.T  # Return as [channels, time]

class AdvancedSAI:
    """Advanced SAI implementation with research-grade algorithms."""
    
    def __init__(self, num_channels=71, sai_width=400, fs=22050):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.fs = fs
        
        # SAI parameters based on research
        self.trigger_window_width = min(400, sai_width)
        self.future_lags = sai_width // 4
        self.num_triggers_per_frame = 4
        
        # Calculate buffer size
        self.buffer_width = sai_width + int(
            (1 + (self.num_triggers_per_frame - 1) / 2) * self.trigger_window_width
        )
        
        # Initialize buffers
        self.input_buffer = np.zeros((num_channels, self.buffer_width))
        self.output_buffer = np.zeros((num_channels, sai_width))
        
        # Create proper Hann-like window for trigger detection
        self.window = np.sin(np.linspace(
            np.pi / self.trigger_window_width, np.pi, self.trigger_window_width
        )) ** 2  # Squared sine window
        
        print(f"SAI initialized: {sai_width} lags, {num_channels} channels")
        
    def process_segment(self, nap_segment):
        """Process NAP segment through advanced SAI algorithm."""
        self._update_input_buffer(nap_segment)
        return self._compute_sai_frame()
        
    def _update_input_buffer(self, nap_segment):
        """Efficiently update the circular input buffer."""
        segment_width = nap_segment.shape[1]
        
        if segment_width >= self.buffer_width:
            # Input larger than buffer - take most recent samples
            self.input_buffer = nap_segment[:, -self.buffer_width:]
        else:
            # Shift existing data and append new samples
            shift_amount = segment_width
            self.input_buffer[:, :-shift_amount] = self.input_buffer[:, shift_amount:]
            self.input_buffer[:, -shift_amount:] = nap_segment
            
    def _compute_sai_frame(self):
        """Compute SAI frame using proper correlation algorithm."""
        self.output_buffer.fill(0.0)
        
        num_samples = self.input_buffer.shape[1]
        window_hop = self.trigger_window_width // 2
        
        # Calculate window positioning
        last_window_start = num_samples - self.trigger_window_width
        first_window_start = last_window_start - (self.num_triggers_per_frame - 1) * window_hop
        
        window_range_start = first_window_start - self.future_lags
        offset_range_start = first_window_start - self.sai_width + 1
        
        if offset_range_start <= 0 or window_range_start < 0:
            return self.output_buffer.copy()
            
        # Process each channel
        for ch in range(self.num_channels):
            channel_signal = self.input_buffer[ch, :]
            
            # Apply light smoothing for trigger detection only
            smoothed_signal = self._smooth_for_triggers(channel_signal)
            
            # Process each trigger window
            for trigger_idx in range(self.num_triggers_per_frame):
                window_offset = trigger_idx * window_hop
                current_window_start = window_range_start + window_offset
                current_window_end = current_window_start + self.trigger_window_width
                
                # Boundary check
                if current_window_end > num_samples:
                    continue
                    
                # Extract and window the trigger region
                trigger_region = smoothed_signal[current_window_start:current_window_end]
                windowed_trigger = trigger_region * self.window
                
                # Find the strongest trigger point
                peak_value = np.max(windowed_trigger)
                if peak_value > 0:
                    peak_location = np.argmax(windowed_trigger)
                    trigger_time = current_window_start + peak_location
                else:
                    # Fallback to window center
                    trigger_time = current_window_start + len(self.window) // 2
                    peak_value = 0.1
                
                # Calculate correlation segment bounds
                correlation_start = trigger_time - self.sai_width + 1 + offset_range_start
                correlation_end = correlation_start + self.sai_width
                
                # Boundary check for correlation segment
                if correlation_start >= 0 and correlation_end <= num_samples:
                    # Extract correlation segment from original (unsmoothed) signal
                    correlation_segment = channel_signal[correlation_start:correlation_end]
                    
                    # Compute adaptive blending weight
                    blend_weight = self._compute_blend_weight(peak_value, trigger_idx)
                    
                    # Accumulate into SAI output
                    self.output_buffer[ch, :] *= (1.0 - blend_weight)
                    self.output_buffer[ch, :] += blend_weight * correlation_segment
                    
        return self.output_buffer.copy()
        
    def _smooth_for_triggers(self, signal):
        """Apply minimal smoothing for trigger detection."""
        # Very light exponential smoothing
        smoothed = signal.copy()
        alpha = 0.1
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
        
    def _compute_blend_weight(self, peak_value, trigger_index):
        """Compute adaptive blending weight."""
        base_weight = 0.25  # Base contribution per trigger
        peak_boost = 2.0 * peak_value / (1.0 + peak_value)  # Sigmoid boost
        
        # Slightly weight earlier triggers more
        temporal_weight = 1.0 - 0.1 * trigger_index / self.num_triggers_per_frame
        
        final_weight = base_weight * peak_boost * temporal_weight
        return np.clip(final_weight, 0.01, 0.8)

class RealTimeCARFACSAI:
    """Real-time CARFAC-SAI analysis system."""
    
    def __init__(self, chunk_size=512, sample_rate=22050, sai_width=400):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        
        # Initialize CARFAC
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.n_channels = self.carfac.n_channels
        
        # Initialize SAI
        self.sai = AdvancedSAI(
            num_channels=self.n_channels,
            sai_width=sai_width,
            fs=sample_rate
        )
        
        # Audio processing
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Visualization setup
        self._setup_visualization()
        
        # Audio hardware
        self.p = None
        self.stream = None
        
    def _setup_visualization(self):
        """Setup high-quality visualization."""
        # Create figure with good aspect ratio
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.sai_data = np.zeros((self.n_channels, self.sai_width))
        
        # Create beautiful colormap
        cmap = self._create_research_colormap()
        
        # Create image with high quality settings
        self.im = self.ax.imshow(
            self.sai_data,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            animated=True  # For smoother animation
        )
        
        # Professional styling
        self.ax.set_title('Real-time CARFAC Cochlear Model with SAI Analysis', 
                         fontsize=18, fontweight='bold', pad=20)
        self.ax.set_xlabel('Time Lag (ms)', fontsize=14)
        self.ax.set_ylabel('Cochlear Frequency Channel (Hz)', fontsize=14)
        
        # Add informative colorbar
        cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8, aspect=30)
        cbar.set_label('Temporal Correlation Strength', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Set up frequency axis with actual CARFAC frequencies
        n_freq_ticks = 12
        freq_indices = np.linspace(0, len(self.carfac.pole_freqs)-1, n_freq_ticks, dtype=int)
        freq_labels = [f'{float(self.carfac.pole_freqs[i]):.0f}' for i in freq_indices]
        self.ax.set_yticks(freq_indices)
        self.ax.set_yticklabels(freq_labels)
        
        # Set up time lag axis
        max_lag_ms = self.sai_width * 1000.0 / self.sample_rate
        n_time_ticks = 12
        time_positions = np.linspace(0, self.sai_width-1, n_time_ticks)
        time_labels = [f'{pos * max_lag_ms / (self.sai_width-1):.1f}' 
                      for pos in time_positions]
        self.ax.set_xticks(time_positions)
        self.ax.set_xticklabels(time_labels)
        
        # Add subtle grid
        self.ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Improve layout
        plt.tight_layout()
        
        # Add text annotation for pitch interpretation
        self.pitch_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        
    def _create_research_colormap(self):
        """Create research-quality colormap."""
        n_colors = 2048  # Very high resolution
        
        # Advanced color progression
        hues = np.linspace(0.8, 0.0, n_colors)  # Blue through red
        sats = 0.85 * np.ones(n_colors)  # Consistent saturation
        vals = np.power(np.linspace(0.05, 1.0, n_colors), 0.8)  # Perceptual brightness
        
        # Create HSV array and convert to RGB
        hsv_array = np.stack([hues, sats, vals], axis=-1)
        rgb_array = hsv_to_rgb(hsv_array.reshape(-1, 1, 3)).reshape(-1, 3)
        
        return ListedColormap(rgb_array)
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """High-quality audio input callback."""
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Convert audio data
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Apply input conditioning
            audio_data = np.clip(audio_data * 2.5, -1.0, 1.0)  # Boost and limit
            
            # Queue for processing
            self.audio_queue.put_nowait(audio_data)
            
        except queue.Full:
            pass  # Drop samples if processing can't keep up
        except Exception as e:
            print(f"Audio callback error: {e}")
            
        return (in_data, pyaudio.paContinue)
        
    def process_audio(self):
        """Main audio processing loop."""
        print("Audio processing thread started...")
        
        while self.running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process through CARFAC (this is where the magic happens!)
                nap_output = self.carfac.process_chunk(audio_chunk)
                
                # Process through SAI
                sai_frame = self.sai.process_segment(nap_output)
                
                # Post-process for visualization
                self.sai_data = self._enhance_for_display(sai_frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                continue
                
    def _enhance_for_display(self, sai_frame):
        """Enhance SAI data for optimal visualization."""
        # Temporal smoothing for stability
        temporal_alpha = 0.3
        if hasattr(self, '_previous_sai'):
            sai_frame = temporal_alpha * sai_frame + (1 - temporal_alpha) * self._previous_sai
        self._previous_sai = sai_frame.copy()
        
        # Dynamic range enhancement
        sai_frame = np.power(np.abs(sai_frame), 0.75)
        
        # Adaptive normalization
        percentile_98 = np.percentile(sai_frame, 98)
        if percentile_98 > 1e-6:
            sai_frame = np.clip(sai_frame / percentile_98, 0, 1)
            
        return sai_frame
        
    def _analyze_pitch_content(self, sai_data):
        """Analyze and display pitch information."""
        if sai_data.size == 0:
            return ""
            
        # Find strongest correlations
        mean_across_channels = np.mean(sai_data, axis=0)
        max_lag_idx = np.argmax(mean_across_channels)
        max_correlation = mean_across_channels[max_lag_idx]
        
        if max_correlation > 0.3:  # Significant correlation
            lag_time_ms = max_lag_idx * 1000.0 / self.sample_rate
            if lag_time_ms > 2.0:  # Reasonable pitch range
                fundamental_freq = 1000.0 / lag_time_ms
                return f"Pitch: {fundamental_freq:.1f} Hz ({lag_time_ms:.1f} ms period)"
        
        return "No clear pitch detected"
        
    def update_visualization(self, frame):
        """Update the visualization display."""
        if hasattr(self, 'sai_data'):
            # Update image
            self.im.set_array(self.sai_data)
            
            # Update title with timestamp and stats
            max_corr = np.max(self.sai_data)
            mean_corr = np.mean(self.sai_data)
            
            self.ax.set_title(
                f'CARFAC Cochlear Analysis - {time.strftime("%H:%M:%S")} '
                f'(Peak: {max_corr:.3f}, Mean: {mean_corr:.3f})',
                fontsize=18, fontweight='bold'
            )
            
            # Update pitch information
            pitch_info = self._analyze_pitch_content(self.sai_data)
            self.pitch_text.set_text(pitch_info)
        
        return [self.im, self.pitch_text]
        
    def start(self):
        """Start the real-time analysis system."""
        print("\n" + "="*60)
        print("REAL-TIME CARFAC COCHLEAR ANALYSIS WITH SAI")
        print("="*60)
        print("Using research-grade cochlear model from CARFAC-JAX")
        print("Speak, sing, play instruments, or whistle to see:")
        print("  • Cochlear frequency responses")
        print("  • Pitch periods as vertical lines")
        print("  • Temporal correlation patterns")
        print("  • Real-time pitch detection")
        print("="*60)
        
        try:
            # Initialize audio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
            
            # Start processing
            self.running = True
            self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.audio_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            print(f"Audio stream started: {self.sample_rate}Hz, {self.chunk_size} samples/chunk")
            
            # Start visualization
            self.animation = animation.FuncAnimation(
                self.fig,
                self.update_visualization,
                interval=25,  # 40 FPS
                blit=False,
                cache_frame_data=False
            )
            
            plt.show()
            
        except Exception as e:
            print(f"Error starting system: {e}")
            self.stop()
            
    def stop(self):
        """Cleanly stop the system."""
        print("Stopping CARFAC-SAI analysis...")
        
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
            
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
            
        plt.close('all')
        print("System stopped.")

def main():
    """Main entry point."""
    print("Initializing Real-time CARFAC-SAI Analysis System...")
    
    try:
        # Create the system
        system = RealTimeCARFACSAI(
            chunk_size=512,     # Smaller chunks for lower latency
            sample_rate=22050,  # Standard rate for CARFAC
            sai_width=400       # Good resolution for pitch analysis
        )
        
        # Start the analysis
        system.start()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'system' in locals():
            system.stop()
        print("Goodbye!")

if __name__ == "__main__":
    main()