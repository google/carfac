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

sys.path.append('./jax')  
import jax
import jax.numpy as jnp
import carfac

class RealCARFACProcessor:    
    def __init__(self, fs=22050, n_channels=71):
        self.fs = fs
                
        self.params = carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(self.params)
        
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.pole_freqs = self.hypers.ears[0].pole_freqs
                
        self.run_segment_jit = jax.jit(
            carfac.run_segment,
            static_argnames=['hypers', 'open_loop']
        )
        
    def process_chunk(self, audio_chunk):
        if len(audio_chunk.shape) == 1:
            audio_input = audio_chunk.reshape(-1, 1)  
        else:
            audio_input = audio_chunk
            
        audio_jax = jnp.array(audio_input, dtype=jnp.float32)
        
        naps, _, self.state, bm_output, _, _ = self.run_segment_jit(
            audio_jax, self.hypers, self.weights, self.state, open_loop=False
        )
        
        nap_output = np.array(naps[:, 0, :])  
        return nap_output.T  

class AdvancedSAI:    
    def __init__(self, num_channels=71, sai_width=400, fs=22050):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.fs = fs
        
        self.trigger_window_width = min(400, sai_width)
        self.future_lags = sai_width // 4
        self.num_triggers_per_frame = 4
        
        self.buffer_width = sai_width + int(
            (1 + (self.num_triggers_per_frame - 1) / 2) * self.trigger_window_width
        )
        
        self.input_buffer = np.zeros((num_channels, self.buffer_width))
        self.output_buffer = np.zeros((num_channels, sai_width))
        
        self.window = np.sin(np.linspace(
            np.pi / self.trigger_window_width, np.pi, self.trigger_window_width
        )) ** 2 
        
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
        
        final_weight = base_weight * peak_boost * temporal_weight
        return np.clip(final_weight, 0.01, 0.8)

class RealTimeCARFACSAI:    
    def __init__(self, chunk_size=512, sample_rate=22050, sai_width=400):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.n_channels = self.carfac.n_channels
        
        self.sai = AdvancedSAI(
            num_channels=self.n_channels,
            sai_width=sai_width,
            fs=sample_rate
        )
        
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
        self._setup_visualization()
        
        self.p = None
        self.stream = None
        
    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.display_width = min(300, self.sai_width)
        self.sai_data = np.zeros((self.n_channels, self.display_width))
        
        cmap = self._create_research_colormap()
        
        self.im = self.ax.imshow(
            self.sai_data,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            animated=True  
        )
        
        
        self.ax.axis("off")

        plt.tight_layout()
        
        self.pitch_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        
    def _create_research_colormap(self):
        n_colors = self.n_channels
        hues = np.linspace(0.0, 0.8, n_colors) 
        sats = np.ones(n_colors) * 0.85         
        vals = np.ones(n_colors)               
        hsv_array = np.stack([hues, sats, vals], axis=-1)
        rgb_array = hsv_to_rgb(hsv_array.reshape(-1, 1, 3)).reshape(-1, 3)
        return ListedColormap(rgb_array)
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            audio_data = np.clip(audio_data * 2.5, -1.0, 1.0) 
            
            self.audio_queue.put_nowait(audio_data)
            
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
                
            except queue.Empty:
                continue
                
    def _enhance_for_display(self, sai_frame):

        temporal_alpha = 0.3
        if hasattr(self, '_previous_sai'):
            sai_frame = temporal_alpha * sai_frame + (1 - temporal_alpha) * self._previous_sai
        self._previous_sai = sai_frame.copy()

        sai_frame = np.power(np.abs(sai_frame), 0.75)

        for ch in range(sai_frame.shape[0]):
            max_val = np.max(sai_frame[ch, :])
            if max_val > 1e-6:
                sai_frame[ch, :] = sai_frame[ch, :] / max_val

        return sai_frame[:, :self.display_width]

        
    def _analyze_pitch_content(self, sai_data):
        if sai_data.size == 0:
            return ""
            
        mean_across_channels = np.mean(sai_data, axis=0)
        max_lag_idx = np.argmax(mean_across_channels)
        max_correlation = mean_across_channels[max_lag_idx]
        
        if max_correlation > 0.3: 
            lag_time_ms = max_lag_idx * 1000.0 / self.sample_rate
            if lag_time_ms > 2.0: 
                fundamental_freq = 1000.0 / lag_time_ms
                return f"{fundamental_freq:.1f} Hz"
        
        return "No clear pitch detected"
        
    def update_visualization(self, frame):
        if hasattr(self, 'sai_data'):
            display_width = min(300, self.sai_width)
            self.im.set_data(self.sai_data[:, :display_width])
            self.im.set_array(self.sai_data)
            
            max_corr = np.max(self.sai_data)
            mean_corr = np.mean(self.sai_data)
            
            pitch_info = self._analyze_pitch_content(self.sai_data)
            self.pitch_text.set_text(pitch_info)
        
        return [self.im, self.pitch_text]
    
    def start(self):
        
        try:
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
            
            self.running = True
            self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.audio_thread.start()
            
            self.stream.start_stream()
            
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
            self.stop()
            
    def stop(self):
        
        self.running = False
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
            
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
            
        plt.close('all')

def main():

        system = RealTimeCARFACSAI(
            chunk_size=512,    
            sample_rate=22050,  
            sai_width=400       
        )
        
        system.start()

if __name__ == "__main__":
    main()