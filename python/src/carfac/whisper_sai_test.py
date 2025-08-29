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

class RealTimeSAIWhisper:
    def __init__(self, chunk_size=512, sample_rate=22050, sai_width=400, whisper_model="medium", colormap_style="enhanced"):
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

        # Whisper
        self.audio_model = whisper.load_model(whisper_model)
        self.data_queue = queue.Queue()
        self.transcription_lines = ['']
        self.transcription_lock = threading.Lock()

        # Dynamic range tracking for better contrast
        self.intensity_history = []
        self.max_history_length = 100

        self._setup_visualization()
        self.p = None
        self.stream = None

    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.display_width = min(300, self.sai_width)
        
        self.temporal_buffer_width = 200  # Number of time frames to display
        self.sai_temporal_buffer = np.zeros((self.n_channels, self.temporal_buffer_width))
        self.frame_counter = 0

        cmap = self._create_enhanced_colormap(self.colormap_style)
        self.im = self.ax.imshow(
            self.sai_temporal_buffer, aspect='auto', origin='lower',
            cmap=cmap, interpolation='bilinear', vmin=0, vmax=1, animated=True,
            extent=[0, self.temporal_buffer_width, 0, self.n_channels]
        )

        self.ax.set_xlabel('Time (frames →)', fontsize=12)
        self.ax.set_ylabel('Frequency Channel', fontsize=12)
        self.ax.set_title('Real-time SAI Visualization (Time flows left → right)', fontsize=14)

        self.cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8, pad=0.02)
        self.cbar.set_label('Correlation Strength', fontsize=10)

        self.pitch_text = self.ax.text(
            0.98, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', horizontalalignment='right', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
        )

        self.transcription_text = self.ax.text(
            0.02, 0.02, '', transform=self.ax.transAxes,
            verticalalignment='bottom', fontsize=12,
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9)
        )

        self._add_frequency_labels()
        
        self._add_time_indicators()
        
        plt.tight_layout()

    def _add_frequency_labels(self):
        freq_labels = []
        n_labels = 8
        for i in range(n_labels):
            channel_idx = int(i * (self.n_channels - 1) / (n_labels - 1))
            freq = 20000 * (0.02 ** (channel_idx / (self.n_channels - 1)))
            if freq >= 1000:
                freq_labels.append(f'{freq/1000:.1f}k')
            else:
                freq_labels.append(f'{int(freq)}')
        
        self.ax.set_yticks(np.linspace(0, self.n_channels-1, n_labels))
        self.ax.set_yticklabels(freq_labels)
    
    def _add_time_indicators(self):
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
            n_colors = self.n_channels
            colors = []
            for i in range(n_colors):
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

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_data = np.clip(audio_data * 2.5, -1.0, 1.0)
            self.audio_queue.put_nowait(audio_data)
            self.data_queue.put_nowait((audio_data * 32768).astype(np.int16).tobytes())
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                nap_output = self.carfac.process_chunk(audio_chunk)
                sai_frame = self.sai.process_segment(nap_output)
                self.sai_data = self._enhance_for_display(sai_frame)
                self.frame_counter += 1
            except queue.Empty:
                continue

    def _enhance_for_display(self, sai_frame):
        """Enhanced processing with temporal scrolling and dynamic range adaptation"""
        temporal_alpha = 0.3
        if hasattr(self, '_previous_sai'):
            sai_frame = temporal_alpha * sai_frame + (1 - temporal_alpha) * self._previous_sai
        self._previous_sai = sai_frame.copy()
        
        sai_frame = np.power(np.abs(sai_frame), 0.6)
        
        current_max = np.max(sai_frame)
        if current_max > 0:
            self.intensity_history.append(current_max)
            if len(self.intensity_history) > self.max_history_length:
                self.intensity_history.pop(0)
            
            if len(self.intensity_history) > 10:
                adaptive_max = np.percentile(self.intensity_history, 95)
                sai_frame = np.clip(sai_frame / adaptive_max, 0, 1)
            else:
                sai_frame = sai_frame / current_max
        
        for ch in range(sai_frame.shape[0]):
            ch_max = np.max(sai_frame[ch, :])
            if ch_max > 0.1:  # Only normalize channels with significant activity
                sai_frame[ch, :] = sai_frame[ch, :] / ch_max
        
        from scipy import ndimage
        sai_frame = ndimage.gaussian_filter(sai_frame, sigma=0.5)
        
        temporal_summary = np.max(sai_frame, axis=1)  # Shape: (n_channels,)
        
        self.sai_temporal_buffer[:, :-1] = self.sai_temporal_buffer[:, 1:]  # Scroll left
        self.sai_temporal_buffer[:, -1] = temporal_summary  # Add new data on right
        
        return self.sai_temporal_buffer

    def _analyze_pitch_content(self, sai_data):
        if sai_data.size == 0:
            return ""
        
        current_frame = sai_data[:, -1]  # Get the latest temporal frame
        
        channel_weights = np.linspace(0.5, 1.0, len(current_frame))  # Weight lower frequencies more

        if np.max(current_frame) > 0.3:
            max_channel = np.argmax(current_frame)
            max_intensity = current_frame[max_channel]
            
            freq_ratio = max_channel / (len(current_frame) - 1)
            estimated_freq = 20000 * (0.02 ** (1 - freq_ratio))  # Reverse of the frequency mapping
            
            if max_intensity > 0.3:
                return f"~{estimated_freq:.0f} Hz (activity: {max_intensity:.2f})"
        
        return "No clear pitch detected"

    def update_visualization(self, frame):
        if hasattr(self, 'sai_data'):
            current_max = np.max(self.sai_data)
            if current_max > 0:
                self.im.set_clim(vmin=0, vmax=min(1.0, current_max * 1.2))
            
            self.im.set_data(self.sai_data)
            pitch_info = self._analyze_pitch_content(self.sai_data)
            self.pitch_text.set_text(pitch_info)

        with self.transcription_lock:
            combined_text = '\n'.join(self.transcription_lines)
        self.transcription_text.set_text(combined_text)

        return [self.im, self.pitch_text, self.transcription_text]

    def whisper_thread(self):
        phrase_bytes = bytes()
        phrase_time = None
        phrase_timeout = 3.0
        while self.running:
            if not self.data_queue.empty():
                now = time.time()
                if phrase_time and now - phrase_time > phrase_timeout:
                    phrase_bytes = bytes()
                phrase_time = now

                audio_data = b''.join(list(self.data_queue.queue))
                self.data_queue.queue.clear()
                phrase_bytes += audio_data

                audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                with self.transcription_lock:
                    self.transcription_lines[-1] = text
            else:
                time.sleep(0.1)

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

        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=25, blit=False
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
        self.colormap_style = style
        new_cmap = self._create_enhanced_colormap(style)
        self.im.set_cmap(new_cmap)
        print(f"Switched to {style} colormap")

if __name__ == "__main__":
    system = RealTimeSAIWhisper(
        chunk_size=512, 
        sample_rate=22050, 
        sai_width=400,
        colormap_style="enhanced"  
    )
    
    system.start()