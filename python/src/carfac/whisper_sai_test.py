#! python3.7
import sys
import os
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb, ListedColormap, LinearSegmentedColormap
import threading
import queue
import time
import torch
import whisper
import speech_recognition as sr

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac

# -------------------- CARFAC + SAI Classes --------------------

class RealCARFACProcessor:
    def __init__(self, fs=22050):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(carfac.CarfacDesignParameters(fs=fs, n_ears=1))
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])

    def process_chunk(self, audio_chunk):
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)
        else:
            audio_input = audio_chunk
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        naps, _, self.state, _, _, _ = self.run_segment_jit(audio_jax, self.hypers, self.weights, self.state, open_loop=False)
        return np.array(naps[:, 0, :]).T

class AdvancedSAI:
    def __init__(self, num_channels=71, sai_width=400):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.trigger_window_width = min(400, sai_width)
        self.future_lags = sai_width // 4
        self.num_triggers_per_frame = 4
        self.buffer_width = sai_width + int((1 + (self.num_triggers_per_frame - 1) / 2) * self.trigger_window_width)
        self.input_buffer = np.zeros((num_channels, self.buffer_width))
        self.output_buffer = np.zeros((num_channels, sai_width))
        self.window = np.sin(np.linspace(np.pi / self.trigger_window_width, np.pi, self.trigger_window_width)) ** 2
        print(f"SAI initialized: {sai_width} lags, {num_channels} channels")

    def process_segment(self, nap_segment):
        self._update_input_buffer(nap_segment)
        return self._compute_sai_frame()

    def _update_input_buffer(self, nap_segment):
        segment_width = nap_segment.shape[1]
        if segment_width >= self.buffer_width:
            self.input_buffer = nap_segment[:, -self.buffer_width:]
        else:
            shift_amount = segment_width
            self.input_buffer[:, :-shift_amount] = self.input_buffer[:, shift_amount:]
            self.input_buffer[:, -shift_amount:] = nap_segment

    def _compute_sai_frame(self):
        self.output_buffer.fill(0.0)
        num_samples = self.input_buffer.shape[1]
        window_hop = self.trigger_window_width // 2
        last_window_start = num_samples - self.trigger_window_width
        first_window_start = last_window_start - (self.num_triggers_per_frame - 1) * window_hop
        window_range_start = first_window_start - self.future_lags
        offset_range_start = first_window_start - self.sai_width + 1
        if offset_range_start <= 0 or window_range_start < 0:
            return self.output_buffer.copy()
        for ch in range(self.num_channels):
            channel_signal = self.input_buffer[ch, :]
            smoothed_signal = self._smooth_for_triggers(channel_signal)
            for trigger_idx in range(self.num_triggers_per_frame):
                window_offset = trigger_idx * window_hop
                current_window_start = window_range_start + window_offset
                current_window_end = current_window_start + self.trigger_window_width
                if current_window_end > num_samples:
                    continue
                trigger_region = smoothed_signal[current_window_start:current_window_end]
                windowed_trigger = trigger_region * self.window
                peak_value = np.max(windowed_trigger)
                if peak_value > 0:
                    peak_location = np.argmax(windowed_trigger)
                    trigger_time = current_window_start + peak_location
                else:
                    trigger_time = current_window_start + len(self.window) // 2
                    peak_value = 0.1
                correlation_start = trigger_time - self.sai_width + 1 + offset_range_start
                correlation_end = correlation_start + self.sai_width
                if correlation_start >= 0 and correlation_end <= num_samples:
                    correlation_segment = channel_signal[correlation_start:correlation_end]
                    blend_weight = self._compute_blend_weight(peak_value, trigger_idx)
                    self.output_buffer[ch, :] *= (1.0 - blend_weight)
                    self.output_buffer[ch, :] += blend_weight * correlation_segment
        return self.output_buffer.copy()

    def _smooth_for_triggers(self, signal):
        smoothed = signal.copy()
        alpha = 0.1
        for i in range(1, len(smoothed)):
            smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    def _compute_blend_weight(self, peak_value, trigger_index):
        base_weight = 0.25
        peak_boost = 2.0 * peak_value / (1.0 + peak_value)
        temporal_weight = 1.0 - 0.1 * trigger_index / self.num_triggers_per_frame
        return np.clip(base_weight * peak_boost * temporal_weight, 0.01, 0.8)

# -------------------- Real-time SAI + Whisper --------------------

class RealTimeSAIWhisper:
    def __init__(self, chunk_size=512, sample_rate=22050, sai_width=400, whisper_model="base", colormap_style="enhanced"):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.colormap_style = colormap_style

        # SAI
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.sai = AdvancedSAI(num_channels=self.carfac.n_channels, sai_width=sai_width)
        self.n_channels = self.sai.num_channels

        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False

        # Whisper - use smaller model for better performance
        print(f"Loading Whisper model: {whisper_model}")
        self.audio_model = whisper.load_model(whisper_model)
        print("Whisper model loaded successfully")
        
        self.data_queue = queue.Queue(maxsize=50)  # Larger queue for audio buffering
        self.transcription_lines = ['[Starting transcription...]']
        self.transcription_lock = threading.Lock()

        # Pitch detection - separate from SAI for accuracy
        self.pitch_buffer = np.zeros(chunk_size * 4)  # Buffer for pitch analysis
        self.current_raw_audio = None
        
        # Dynamic range tracking for better contrast
        self.intensity_history = []
        self.max_history_length = 100

        self._setup_visualization()
        self.p = None
        self.stream = None

    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.display_width = min(300, self.sai_width)
        
        # Create scrolling buffer for temporal visualization
        self.temporal_buffer_width = 200  # Number of time frames to display
        # Keep the full SAI lag dimension for proper pitch analysis
        self.sai_temporal_buffer = np.zeros((self.n_channels, self.temporal_buffer_width))
        self.pitch_temporal_buffer = np.zeros(self.temporal_buffer_width)  # Store pitch over time
        self.frame_counter = 0

        # Enhanced colormap selection
        cmap = self._create_enhanced_colormap(self.colormap_style)
        self.im = self.ax.imshow(
            self.sai_temporal_buffer, aspect='auto', origin='lower',
            cmap=cmap, interpolation='bilinear', vmin=0, vmax=1, animated=True,
            extent=[0, self.temporal_buffer_width, 0, self.n_channels]
        )

        # Set up axes labels for temporal view with realistic pitch focus
        self.ax.set_xlabel('Time (frames →)', fontsize=12)
        self.ax.set_ylabel('Frequency Channel (Cochlear CF)', fontsize=12)
        self.ax.set_title('Real-time SAI: Speech/Music Pitch Analysis (50-1000 Hz focus)', fontsize=14)

        # Add colorbar for reference
        self.cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8, pad=0.02)
        self.cbar.set_label('SAI Correlation Strength', fontsize=10)

        # Pitch display (moved to top right, smaller)
        self.pitch_text = self.ax.text(
            0.98, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9)
        )

        # Transcription display - much larger and more prominent
        self.transcription_text = self.ax.text(
            0.02, 0.35, '', transform=self.ax.transAxes,
            verticalalignment='bottom', fontsize=16, fontweight='bold',
            color='black', wrap=True,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.95, edgecolor='navy', linewidth=2)
        )

        # Frequency scale labels
        self._add_frequency_labels()
        
        # Add time indicators
        self._add_time_indicators()
        
        # Store current SAI frame for pitch analysis
        self.current_sai_frame = np.zeros((self.n_channels, self.display_width))
        
        plt.tight_layout()

    def _add_frequency_labels(self):
        """Add frequency labels to show the mapping between channels and frequencies"""
        # Approximate frequency mapping for CARFAC (cochlear model)
        # High frequencies at top, low frequencies at bottom
        freq_labels = []
        n_labels = 8
        for i in range(n_labels):
            channel_idx = int(i * (self.n_channels - 1) / (n_labels - 1))
            # Approximate frequency mapping (CARFAC covers ~20Hz to ~20kHz)
            freq = 20000 * (0.02 ** (channel_idx / (self.n_channels - 1)))
            if freq >= 1000:
                freq_labels.append(f'{freq/1000:.1f}k')
            else:
                freq_labels.append(f'{int(freq)}')
        
        # Add frequency axis
        self.ax.set_yticks(np.linspace(0, self.n_channels-1, n_labels))
        self.ax.set_yticklabels(freq_labels)
    
    def _add_time_indicators(self):
        """Add time scale indicators with realistic pitch information"""
        time_ticks = np.linspace(0, self.temporal_buffer_width, 6)
        time_labels = []
        for tick in time_ticks:
            seconds_ago = (self.temporal_buffer_width - tick) / 40.0  # Approximate frame rate
            if seconds_ago < 0.1:
                time_labels.append('Now')
            else:
                time_labels.append(f'-{seconds_ago:.1f}s')
        
        self.ax.set_xticks(time_ticks)
        self.ax.set_xticklabels(time_labels)
        
        # Add realistic pitch range information
        self.ax.text(0.5, -0.08, 
                    'SAI focuses on speech/music pitch range: 50-1000 Hz (logarithmic lag weighting)', 
                    transform=self.ax.transAxes, ha='center', fontsize=9, style='italic')

    def _create_enhanced_colormap(self, style="enhanced"):
        """Create sophisticated colormaps for different visualization needs"""
        
        if style == "perceptual":
            # Perceptually uniform colormap optimized for auditory data
            colors = [
                '#000033',  # Deep blue (silence)
                '#000080',  # Blue (low activity)
                '#0040FF',  # Bright blue (moderate activity)
                '#00FFFF',  # Cyan (strong activity)
                '#40FF40',  # Green (very strong activity)
                '#FFFF00',  # Yellow (peak activity)
                '#FF8000',  # Orange (very peak activity)
                '#FF0000',  # Red (maximum activity)
                '#FFFFFF'   # White (saturation)
            ]
            return LinearSegmentedColormap.from_list("perceptual_audio", colors, N=256)
            
        elif style == "frequency_specific":
            # Different colors for different frequency bands
            n_colors = self.n_channels
            colors = []
            for i in range(n_colors):
                # Map frequency bands to colors
                freq_ratio = i / (n_colors - 1)
                if freq_ratio < 0.2:  # Very low frequencies - deep red
                    hue = 0.0
                    sat = 0.8 + 0.2 * (freq_ratio / 0.2)
                elif freq_ratio < 0.4:  # Low frequencies - orange to yellow
                    hue = 0.08 + 0.08 * ((freq_ratio - 0.2) / 0.2)
                    sat = 0.9
                elif freq_ratio < 0.6:  # Mid frequencies - green
                    hue = 0.33
                    sat = 0.7 + 0.3 * ((freq_ratio - 0.4) / 0.2)
                elif freq_ratio < 0.8:  # High frequencies - cyan to blue
                    hue = 0.5 + 0.17 * ((freq_ratio - 0.6) / 0.2)
                    sat = 0.8
                else:  # Very high frequencies - purple
                    hue = 0.75 + 0.08 * ((freq_ratio - 0.8) / 0.2)
                    sat = 0.9
                
                val = 0.3 + 0.7 * freq_ratio  # Brightness increases with frequency
                rgb = hsv_to_rgb([[[hue, sat, val]]])[0, 0]
                colors.append(rgb)
            
            return ListedColormap(colors)
            
        elif style == "intensity_enhanced":
            # Enhanced contrast for different intensity levels
            colors = [
                '#000000',  # Black (no activity)
                '#1a0033',  # Very dark purple
                '#330066',  # Dark purple
                '#4d0099',  # Purple
                '#6600cc',  # Bright purple
                '#0066ff',  # Blue
                '#0099ff',  # Light blue
                '#00ccff',  # Cyan
                '#00ffcc',  # Turquoise
                '#00ff99',  # Green-cyan
                '#33ff66',  # Green
                '#66ff33',  # Yellow-green
                '#99ff00',  # Yellow
                '#ccff00',  # Bright yellow
                '#ffcc00',  # Orange
                '#ff9900',  # Bright orange
                '#ff6600',  # Red-orange
                '#ff3300',  # Red
                '#ff0066',  # Pink-red
                '#ffffff'   # White (maximum)
            ]
            return LinearSegmentedColormap.from_list("intensity_enhanced", colors, N=256)
            
        else:  # "enhanced" - default improved version
            # Multi-hue colormap with better discrimination
            colors = [
                '#000022',  # Very dark blue
                '#000055',  # Dark blue
                '#0033AA',  # Blue
                '#0066FF',  # Bright blue
                '#00AAFF',  # Sky blue
                '#00FFAA',  # Cyan-green
                '#33FF77',  # Green
                '#77FF33',  # Yellow-green
                '#AAFF00',  # Yellow
                '#FFAA00',  # Orange
                '#FF7700',  # Dark orange
                '#FF3300',  # Red
                '#FF0044',  # Pink-red
                '#CC0077',  # Magenta
                '#FFFFFF'   # White
            ]
            return LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)

    # ---------------- Audio callback ----------------
    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_data = np.clip(audio_data * 2.5, -1.0, 1.0)
            self.audio_queue.put_nowait(audio_data)
            self.data_queue.put_nowait((audio_data * 32768).astype(np.int16).tobytes())
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    # ---------------- Audio processing ----------------
    def process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Store raw audio for pitch analysis
                self.current_raw_audio = audio_chunk.copy()
                
                # Update pitch buffer for continuous pitch tracking
                self._update_pitch_buffer(audio_chunk)
                
                # Process through CARFAC and SAI for visualization
                nap_output = self.carfac.process_chunk(audio_chunk)
                sai_frame = self.sai.process_segment(nap_output)
                self.sai_data = self._enhance_for_display(sai_frame)
                self.frame_counter += 1
            except queue.Empty:
                continue
                
    def _update_pitch_buffer(self, new_audio):
        """Update rolling buffer for pitch analysis"""
        # Shift existing audio left and add new audio on right
        shift_amount = len(new_audio)
        self.pitch_buffer[:-shift_amount] = self.pitch_buffer[shift_amount:]
        self.pitch_buffer[-shift_amount:] = new_audio

    def _enhance_for_display(self, sai_frame):
        """Enhanced processing with temporal scrolling and logarithmic lag handling"""
        temporal_alpha = 0.3
        if hasattr(self, '_previous_sai'):
            sai_frame = temporal_alpha * sai_frame + (1 - temporal_alpha) * self._previous_sai
        self._previous_sai = sai_frame.copy()
        
        # Store the ORIGINAL SAI frame for pitch analysis (before any processing)
        self.current_sai_frame = sai_frame.copy()
        
        # Apply gamma correction for better perception (only for display)
        display_frame = np.power(np.abs(sai_frame), 0.6)
        
        # Dynamic range adaptation
        current_max = np.max(display_frame)
        if current_max > 0:
            self.intensity_history.append(current_max)
            if len(self.intensity_history) > self.max_history_length:
                self.intensity_history.pop(0)
            
            # Use percentile-based normalization for better contrast
            if len(self.intensity_history) > 10:
                adaptive_max = np.percentile(self.intensity_history, 95)
                display_frame = np.clip(display_frame / adaptive_max, 0, 1)
            else:
                display_frame = display_frame / current_max
        
        # Channel-wise normalization for better frequency discrimination
        for ch in range(display_frame.shape[0]):
            ch_max = np.max(display_frame[ch, :])
            if ch_max > 0.1:  # Only normalize channels with significant activity
                display_frame[ch, :] = display_frame[ch, :] / ch_max
        
        # Apply slight spatial smoothing to reduce noise while preserving structure
        from scipy import ndimage
        display_frame = ndimage.gaussian_filter(display_frame, sigma=0.5)
        
        # For temporal display: create logarithmic summary across lag dimension
        # SAI lag axis is inherently logarithmic - sample different lag regions with log spacing
        lag_samples = self._get_log_lag_samples(display_frame.shape[1])
        
        # Extract values at logarithmically spaced lag points and weight by importance
        log_weighted_summary = np.zeros(display_frame.shape[0])
        
        for ch in range(display_frame.shape[0]):
            # Sample at log-spaced lag indices
            lag_values = display_frame[ch, lag_samples]
            
            # Weight shorter lags more (they correspond to more salient pitches)
            lag_weights = np.exp(-np.arange(len(lag_samples)) * 0.1)
            log_weighted_summary[ch] = np.sum(lag_values * lag_weights) / np.sum(lag_weights)
        
        # Update temporal buffer by scrolling left and adding new data on the right
        self.sai_temporal_buffer[:, :-1] = self.sai_temporal_buffer[:, 1:]  # Scroll left
        self.sai_temporal_buffer[:, -1] = log_weighted_summary  # Add new data on right
        
        return self.sai_temporal_buffer
    
    def _get_log_lag_samples(self, max_lag):
        """Generate logarithmically spaced lag sample points for realistic pitch ranges"""
        # Focus on human speech and music pitch ranges:
        # Speech: ~80-300 Hz (fundamental)
        # Music: ~50-2000 Hz (instruments)
        # Singing: ~80-1000 Hz
        
        # Convert realistic frequencies to lag samples
        max_realistic_freq = 1000  # 1 kHz - high singing/instruments
        min_realistic_freq = 50    # 50 Hz - very low bass
        
        min_lag = int(self.sample_rate / max_realistic_freq)  # ~22 samples for 1000 Hz
        max_lag_realistic = int(self.sample_rate / min_realistic_freq)  # ~441 samples for 50 Hz
        
        # Don't exceed the actual SAI width
        max_lag_realistic = min(max_lag_realistic, max_lag - 1, 400)
        
        # Ensure we have a reasonable range
        min_lag = max(min_lag, 1)
        
        if max_lag_realistic > min_lag:
            # Generate log-spaced points in the realistic pitch range
            log_points = np.logspace(np.log10(min_lag), np.log10(max_lag_realistic), num=15)
            lag_indices = np.unique(np.round(log_points).astype(int))
            # Ensure we don't exceed array bounds
            lag_indices = lag_indices[lag_indices < max_lag]
        else:
            # Fallback to a few short lags
            lag_indices = np.array([min_lag, min_lag*2, min_lag*4])
            lag_indices = lag_indices[lag_indices < max_lag]
            
        return lag_indices

    # ---------------- Pitch analysis ----------------
    def _analyze_pitch_content(self, sai_data):
        """Hybrid pitch detection: Fast autocorrelation + SAI validation"""
        if self.current_raw_audio is None or len(self.pitch_buffer) == 0:
            return "No audio data"
        
        # Method 1: Fast autocorrelation pitch detection on raw audio
        pitch_hz, confidence = self._fast_pitch_detection(self.pitch_buffer)
        
        # Method 2: SAI-based validation (optional, for comparison)
        sai_pitch_info = self._sai_pitch_validation()
        
        # Format the output
        if confidence > 0.5:
            confidence_stars = "★★★" if confidence > 0.8 else "★★" if confidence > 0.65 else "★"
            return f"{pitch_hz:.1f} Hz {confidence_stars} (conf: {confidence:.2f}) {sai_pitch_info}"
        else:
            return f"Weak signal: {pitch_hz:.1f} Hz (conf: {confidence:.2f}) {sai_pitch_info}"
    
    def _fast_pitch_detection(self, audio_signal):
        """Fast, accurate pitch detection using autocorrelation"""
        # Remove DC component
        audio_signal = audio_signal - np.mean(audio_signal)
        
        # Apply window to reduce edge effects
        window = np.hanning(len(audio_signal))
        windowed_signal = audio_signal * window
        
        # Compute autocorrelation using FFT (much faster than direct computation)
        autocorr = np.correlate(windowed_signal, windowed_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        
        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return 0.0, 0.0
        
        # Define pitch search range
        min_period = int(self.sample_rate / 800)  # 800 Hz max
        max_period = int(self.sample_rate / 50)   # 50 Hz min
        max_period = min(max_period, len(autocorr) - 1)
        
        if min_period >= max_period:
            return 0.0, 0.0
        
        # Find peaks in the autocorrelation
        search_region = autocorr[min_period:max_period]
        
        if len(search_region) == 0:
            return 0.0, 0.0
        
        # Find the highest peak
        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]
        period_samples = peak_idx + min_period
        
        # Calculate pitch frequency
        pitch_hz = self.sample_rate / period_samples
        
        # Calculate confidence based on peak height and clarity
        noise_floor = np.percentile(search_region, 10)  # Bottom 10% as noise
        peak_prominence = (peak_value - noise_floor) / (1.0 - noise_floor + 1e-6)
        
        # Look for harmonic confirmation
        harmonic_score = 0
        for harmonic in [2, 3]:
            harmonic_period = period_samples // harmonic
            if harmonic_period >= min_period and harmonic_period < len(autocorr):
                harmonic_strength = autocorr[harmonic_period]
                harmonic_score += harmonic_strength
        
        # Combined confidence
        confidence = peak_prominence * (1 + 0.1 * harmonic_score)
        confidence = np.clip(confidence, 0, 1)
        
        return pitch_hz, confidence
    
    def _sai_pitch_validation(self):
        """Optional SAI-based pitch info for comparison/validation"""
        if not hasattr(self, 'current_sai_frame') or self.current_sai_frame.size == 0:
            return ""
        
        # Quick SAI analysis for validation
        sai_frame = self.current_sai_frame
        summed_sai = np.sum(sai_frame, axis=0)
        
        if np.max(summed_sai) > 0.01:
            # Find rough SAI-based pitch for comparison
            min_lag = int(self.sample_rate / 400)
            max_lag = int(self.sample_rate / 80)
            max_lag = min(max_lag, len(summed_sai) - 1)
            
            if min_lag < max_lag:
                sai_region = summed_sai[min_lag:max_lag]
                if len(sai_region) > 0:
                    peak_idx = np.argmax(sai_region)
                    sai_period = peak_idx + min_lag
                    sai_pitch = self.sample_rate / sai_period
                    sai_strength = np.max(sai_region)
                    return f"[SAI: {sai_pitch:.0f}Hz]"
        
        return "[SAI: weak]"

    # ---------------- Visualization ----------------
    def update_visualization(self, frame):
        if hasattr(self, 'sai_data'):
            # Update with dynamic range adjustment
            current_max = np.max(self.sai_data)
            if current_max > 0:
                self.im.set_clim(vmin=0, vmax=min(1.0, current_max * 1.2))
            
            self.im.set_data(self.sai_data)
            pitch_info = self._analyze_pitch_content(self.sai_data)
            self.pitch_text.set_text(pitch_info)

        # Update transcription with better formatting for larger display
        with self.transcription_lock:
            # Format transcription with line breaks for better readability
            formatted_lines = []
            for line in self.transcription_lines:
                # Word wrap long lines to fit better in the display
                if len(line) > 60:  # If line is too long, split it
                    words = line.split(' ')
                    current_line = ""
                    for word in words:
                        if len(current_line + word) < 60:
                            current_line += word + " "
                        else:
                            if current_line:
                                formatted_lines.append(current_line.strip())
                            current_line = word + " "
                    if current_line:
                        formatted_lines.append(current_line.strip())
                else:
                    formatted_lines.append(line)
            
            # Take only the most recent lines to fit in display
            display_lines = formatted_lines[-6:]  # Show last 6 lines
            combined_text = '\n'.join(display_lines)
            
        self.transcription_text.set_text(combined_text)

        return [self.im, self.pitch_text, self.transcription_text]

    # ---------------- Whisper transcription thread ----------------
    def whisper_thread(self):
        """Improved Whisper transcription with debugging"""
        phrase_buffer = []
        last_transcription_time = time.time()
        transcription_interval = 2.0  # Transcribe every 2 seconds
        min_audio_length = 1.0  # Minimum 1 second of audio before transcribing
        
        print("Whisper thread started - listening for speech...")
        
        while self.running:
            try:
                # Collect audio data
                audio_collected = False
                if not self.data_queue.empty():
                    # Get all available audio data
                    audio_chunks = []
                    chunks_collected = 0
                    while not self.data_queue.empty():
                        try:
                            chunk = self.data_queue.get_nowait()
                            audio_chunks.append(chunk)
                            chunks_collected += 1
                        except queue.Empty:
                            break
                    
                    # Combine chunks
                    if audio_chunks:
                        combined_audio = b''.join(audio_chunks)
                        phrase_buffer.append(combined_audio)
                        audio_collected = True
                        
                        # Debug: Print audio collection info
                        audio_length = len(combined_audio) / 2 / self.sample_rate  # bytes to seconds
                        print(f"Audio collected: {chunks_collected} chunks, {audio_length:.2f} seconds")
                
                # Check if it's time to transcribe
                current_time = time.time()
                if current_time - last_transcription_time >= transcription_interval:
                    if phrase_buffer:
                        print("Attempting transcription...")
                        
                        # Combine all buffered audio
                        full_audio_bytes = b''.join(phrase_buffer)
                        
                        # Convert to numpy array
                        audio_np = np.frombuffer(full_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        # Check if we have enough audio
                        audio_duration = len(audio_np) / self.sample_rate
                        print(f"Total audio duration: {audio_duration:.2f} seconds")
                        print(f"Audio level (RMS): {np.sqrt(np.mean(audio_np**2)):.4f}")
                        
                        if audio_duration >= min_audio_length:
                            try:
                                print("Running Whisper transcription...")
                                
                                # Transcribe with Whisper
                                result = self.audio_model.transcribe(
                                    audio_np, 
                                    fp16=torch.cuda.is_available(),
                                    language='en',  # Specify language for better performance
                                    task='transcribe',
                                    temperature=0.0,  # More deterministic
                                    no_speech_threshold=0.6,  # Adjust sensitivity
                                    logprob_threshold=-1.0,
                                    compression_ratio_threshold=2.4,
                                    verbose=True  # Add verbose output
                                )
                                
                                text = result['text'].strip()
                                print(f"Whisper result: '{text}'")
                                print(f"Whisper segments: {len(result.get('segments', []))}")
                                
                                # Check for no_speech_prob in segments
                                if 'segments' in result:
                                    for i, segment in enumerate(result['segments']):
                                        no_speech_prob = segment.get('no_speech_prob', 0)
                                        print(f"Segment {i}: no_speech_prob = {no_speech_prob:.3f}")
                                
                                # Update transcription if we got meaningful text
                                if text and len(text) > 1:
                                    with self.transcription_lock:
                                        # Add timestamp and update
                                        timestamp = time.strftime("%H:%M:%S")
                                        new_line = f"[{timestamp}] {text}"
                                        
                                        # Keep only last 3 lines for better readability with larger text
                                        if len(self.transcription_lines) >= 4:
                                            self.transcription_lines = self.transcription_lines[-2:]
                                        
                                        self.transcription_lines.append(new_line)
                                        print(f"✓ Transcribed: {text}")
                                else:
                                    print("No meaningful text detected by Whisper")
                                
                            except Exception as e:
                                print(f"Whisper transcription error: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"Not enough audio: {audio_duration:.2f}s < {min_audio_length}s")
                        
                        # Clear buffer and reset timer
                        phrase_buffer = []
                        last_transcription_time = current_time
                    else:
                        print("No audio in buffer to transcribe")
                        # No audio in buffer, just reset timer
                        last_transcription_time = current_time
                
                elif audio_collected:
                    # Show that we're collecting audio
                    time_until_transcribe = transcription_interval - (current_time - last_transcription_time)
                    print(f"Audio buffering... transcribing in {time_until_transcribe:.1f}s")
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"Whisper thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

    # ---------------- Start / Stop ----------------
    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32, channels=1,
            rate=self.sample_rate, input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback, start=False
        )

        self.running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
        threading.Thread(target=self.whisper_thread, daemon=True).start()
        self.stream.start_stream()

        # Fix animation warning by specifying save_count and cache settings
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=25, blit=False,
            save_count=100, cache_frame_data=False
        )
        plt.show()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        plt.close('all')

    def switch_colormap(self, style):
        """Dynamically switch between different colormap styles"""
        self.colormap_style = style
        new_cmap = self._create_enhanced_colormap(style)
        self.im.set_cmap(new_cmap)
        print(f"Switched to {style} colormap")

# -------------------- Main --------------------
if __name__ == "__main__":
    # Available colormap styles: "enhanced", "perceptual", "frequency_specific", "intensity_enhanced"
    system = RealTimeSAIWhisper(
        chunk_size=512, 
        sample_rate=22050, 
        sai_width=400,
        whisper_model="base",  # Use "base" model for better performance, or "tiny" for fastest
        colormap_style="enhanced"  # Try different styles: "perceptual", "frequency_specific", "intensity_enhanced"
    )
    
    print("Starting real-time SAI visualization with Whisper transcription...")
    print("Speak into your microphone - transcription will appear every 2 seconds")
    print("Press Ctrl+C to stop")
    
    try:
        # You can switch colormaps during runtime:
        # system.switch_colormap("frequency_specific")
        
        system.start()
    except KeyboardInterrupt:
        print("\nStopping...")
        system.stop()