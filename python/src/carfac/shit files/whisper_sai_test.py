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
from datetime import datetime, timedelta
import speech_recognition as sr

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac

# ---------------- CARFAC Processor ----------------
class RealCARFACProcessor:
    def __init__(self, fs=22050):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
            carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        )
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

# ---------------- Pitchogram ----------------
class RealTimePitchogram:
    def __init__(self, num_channels=71, sai_width=400):
        self.num_channels = num_channels
        self.sai_width = sai_width
        self.output_buffer = np.zeros((num_channels, sai_width))
        self.cgram = np.zeros(num_channels)
        self.vowel_matrix = None

    def set_vowel_matrix(self, vowel_matrix):
        self.vowel_matrix = vowel_matrix

    def run_frame(self, sai_frame):
        masked = sai_frame
        self.output_buffer = masked.mean(axis=1, keepdims=True) * np.ones_like(masked)
        
        if self.vowel_matrix is not None:
            self.cgram = 0.2 * masked.mean(axis=1) + 0.8 * self.cgram
            vowel_coords = self.vowel_matrix @ self.cgram
        return self.output_buffer.copy()

# ---------------- Improved Whisper Handler ----------------
class ImprovedWhisperHandler:
    def __init__(self, model_name="small", energy_threshold=1000, phrase_timeout=3.0):
        print(f"Loading Whisper model: {model_name}")
        self.audio_model = whisper.load_model(model_name)
        self.phrase_timeout = phrase_timeout
        
        # Voice activity detection
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = False
        
        # Phrase management
        self.phrase_bytes = bytearray()
        self.phrase_time = None
        self.transcription_lines = ['Listening...']
        self.transcription_lock = threading.Lock()
        
        # Audio queues
        self.whisper_queue = queue.Queue(maxsize=50)
        self.running = False
        
    def add_audio_chunk(self, audio_chunk):
        try:
            # Convert to int16 for Whisper
            audio_int16 = (audio_chunk * 32768).astype(np.int16)
            self.whisper_queue.put_nowait(audio_int16.tobytes())
        except queue.Full:
            # Drop oldest data if queue is full
            try:
                self.whisper_queue.get_nowait()
                self.whisper_queue.put_nowait(audio_int16.tobytes())
            except queue.Empty:
                pass
    
    def whisper_thread(self):
        while self.running:
            try:
                now = datetime.now()
                audio_chunks = []
                
                # Collect available audio chunks
                while not self.whisper_queue.empty():
                    try:
                        chunk = self.whisper_queue.get_nowait()
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if audio_chunks:
                    # Check for phrase boundary
                    phrase_complete = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
                        self.phrase_bytes = bytearray()
                        phrase_complete = True
                    
                    self.phrase_time = now
                    
                    # Add new audio to phrase buffer
                    for chunk in audio_chunks:
                        self.phrase_bytes.extend(chunk)
                    
                    # Limit buffer size to prevent memory issues
                    max_buffer_seconds = 30
                    max_buffer_bytes = max_buffer_seconds * 22050 * 2  # 16-bit samples
                    if len(self.phrase_bytes) > max_buffer_bytes:
                        # Keep only the most recent audio
                        self.phrase_bytes = self.phrase_bytes[-max_buffer_bytes:]
                    
                    # Convert to numpy for Whisper
                    audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Only transcribe if we have enough audio
                    if len(audio_np) > 22050 * 0.5:  # At least 0.5 seconds
                        try:
                            # Transcribe with improved settings
                            result = self.audio_model.transcribe(
                                audio_np, 
                                fp16=torch.cuda.is_available(),
                                language='en',  # Specify language for better accuracy
                                task='transcribe',
                                temperature=0.0,  # More deterministic output
                                no_speech_threshold=0.6,  # Filter out non-speech
                                logprob_threshold=-1.0
                            )
                            
                            text = result['text'].strip()
                            
                            # Update transcription
                            with self.transcription_lock:
                                if phrase_complete and self.transcription_lines[-1]:
                                    self.transcription_lines.append(text)
                                else:
                                    self.transcription_lines[-1] = text
                                    
                                # Limit transcription history
                                if len(self.transcription_lines) > 10:
                                    self.transcription_lines.pop(0)
                                    
                        except Exception as e:
                            print(f"Whisper transcription error: {e}")
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Whisper thread error: {e}")
                time.sleep(0.1)
    
    def get_transcription(self):
        with self.transcription_lock:
            return '\n'.join(self.transcription_lines)
    
    def start(self):
        self.running = True
        threading.Thread(target=self.whisper_thread, daemon=True).start()
    
    def stop(self):
        self.running = False

# ---------------- Real-Time Visualization + Improved Whisper ----------------
class RealTimePitchogramWhisper:
    def __init__(self, chunk_size=512, sample_rate=22050, sai_width=400, whisper_model="small", colormap_style="enhanced"):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.colormap_style = colormap_style

        # CARFAC + Pitchogram
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.pitchogram = RealTimePitchogram(num_channels=self.carfac.n_channels, sai_width=sai_width)
        self.n_channels = self.carfac.n_channels

        # Improved Whisper handler
        self.whisper_handler = ImprovedWhisperHandler(
            model_name=whisper_model,
            energy_threshold=1000,
            phrase_timeout=3.0
        )

        # Audio processing
        self.audio_queue = queue.Queue(maxsize=20)
        self.running = False
        self.intensity_history = []
        self.max_history_length = 100

        # Visualization setup
        self._setup_visualization()
        self.p = None
        self.stream = None

    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.temporal_buffer_width = 200
        self.temporal_buffer = np.zeros((self.n_channels, self.temporal_buffer_width))
        self.frame_counter = 0
        
        cmap = self._create_enhanced_colormap(self.colormap_style)
        self.im = self.ax.imshow(self.temporal_buffer, aspect='auto', origin='lower',
                                 cmap=cmap, interpolation='bilinear', vmin=0, vmax=1,
                                 extent=[0, self.temporal_buffer_width, 0, self.n_channels])
        
       # self.cbar = plt.colorbar(self.im, ax=self.ax, shrink=0.8, pad=0.02)
        
        self.pitch_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', horizontalalignment='right', fontsize=10,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        self.transcription_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                              verticalalignment='bottom', fontsize=11, color='black',
                                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        plt.tight_layout()
        self.ax.axis("off")


    def _create_enhanced_colormap(self, style="enhanced"):
        colors = ['#000022','#000055','#0033AA','#0066FF','#00AAFF','#00FFAA','#33FF77','#77FF33','#AAFF00',
                  '#FFAA00','#FF7700','#FF3300','#FF0044','#CC0077','#FFFFFF']
        return LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_data = np.clip(audio_data * 2.0, -1.0, 1.0)  # Reduced gain
            self.audio_queue.put_nowait(audio_data)
            
            # Send to improved Whisper handler
            self.whisper_handler.add_audio_chunk(audio_data)
            
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                nap_output = self.carfac.process_chunk(audio_chunk)
                pitch_frame = self.pitchogram.run_frame(nap_output)
                
                # Update temporal buffer
                self.temporal_buffer[:, :-1] = self.temporal_buffer[:, 1:]
                self.temporal_buffer[:, -1] = pitch_frame.max(axis=1)
                self.frame_counter += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                continue

    def _analyze_pitch_content(self):
        current_frame = self.temporal_buffer[:, -1]
        if np.max(current_frame) > 0.2:
            max_ch = np.argmax(current_frame)
            freq_ratio = max_ch / max(1, len(current_frame)-1)
            # Improved frequency estimation
            estimated_freq = 80 * (8000/80) ** freq_ratio
            intensity = np.max(current_frame)
            return f"~{estimated_freq:.0f} Hz ({intensity:.2f})"
        return "No clear pitch"

    def update_visualization(self, frame):
        # Update pitchogram display
        current_max = np.max(self.temporal_buffer) if self.temporal_buffer.size else 1.0
        self.im.set_data(self.temporal_buffer)
        self.im.set_clim(vmin=0, vmax=max(0.001, min(1.0, current_max * 1.2)))
        
        # Update pitch analysis
        pitch_info = self._analyze_pitch_content()
        self.pitch_text.set_text(pitch_info)
        
        # Update transcription from improved handler
        transcription_text = self.whisper_handler.get_transcription()
        self.transcription_text.set_text(transcription_text)
        
        return [self.im, self.pitch_text, self.transcription_text]

    def start(self):
        
        # Start Whisper handler
        self.whisper_handler.start()
        
        # Initialize PyAudio
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
        threading.Thread(target=self.process_audio, daemon=True).start()
        self.stream.start_stream()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=50, blit=False
        )
        
        plt.show()

    def stop(self):
        self.running = False
        self.whisper_handler.stop()
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
        plt.close('all')

# ---------------- Run ----------------
if __name__ == "__main__":
    
    try:
        system = RealTimePitchogramWhisper(
            chunk_size=512, 
            sample_rate=22050, 
            sai_width=400,
            whisper_model="large" 
        )
        system.start()
    except KeyboardInterrupt:
        if 'system' in locals():
            system.stop()
    except Exception as e:
        if 'system' in locals():
            system.stop()