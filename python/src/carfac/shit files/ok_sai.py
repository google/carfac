#! python3.7

import sys
import os
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import threading
import queue
import time
import torch
import whisper
from datetime import datetime, timedelta

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac

# ---------------- CARFAC Processor ----------------
class RealCARFACProcessor:
    def __init__(self, fs=16000):
        self.fs = fs
        self.hypers, self.weights, self.state = carfac.design_and_init_carfac(
            carfac.CarfacDesignParameters(fs=fs, n_ears=1)
        )
        self.n_channels = self.hypers.ears[0].car.n_ch
        self.run_segment_jit = jax.jit(carfac.run_segment, static_argnames=['hypers', 'open_loop'])

    def process_chunk(self, audio_chunk):
        # audio_chunk expected as float32 in range [-1,1]
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

# ---------------- Whisper Handler ----------------
class WhisperHandler:
    def __init__(self, model_name="base", non_english=False):
        print(f"Loading Whisper model: {model_name} (non_english={non_english})")
        model = model_name
        if model_name != "large" and not non_english:
            model = model + ".en"
        
        try:
            self.audio_model = whisper.load_model(model)
            self.sample_rate = 16000  # Store sample rate for validation
            print(f"Whisper model '{model}' loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            print("Falling back to 'tiny' model...")
            try:
                self.audio_model = whisper.load_model("tiny")
                self.sample_rate = 16000
                print("Tiny model loaded successfully")
            except Exception as e2:
                print(f"Failed to load any Whisper model: {e2}")
                self.audio_model = None
                self.sample_rate = 16000
        
        self.transcription = []
        self.lock = threading.Lock()
        self.last_transcription_time = time.time()
        self.min_transcription_interval = 1.0  # Minimum time between transcriptions

    def transcribe_audio(self, audio_data, language='en'):
        if self.audio_model is None:
            return None
        
        # Validate input data
        if audio_data is None or len(audio_data) == 0:
            return None
            
        # Convert to numpy array if needed
        if not isinstance(audio_data, np.ndarray):
            try:
                audio_data = np.array(audio_data)
            except:
                return None
        
        # Check minimum length (at least 1 second for reliable transcription)
        min_samples = int(self.sample_rate * 1.0)
        if len(audio_data) < min_samples:
            return None
        
        # Ensure audio is float32 in range [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Check for valid audio data
        if np.all(audio_float == 0) or not np.isfinite(audio_float).all():
            return None
        
        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        elif max_val < 0.001:  # Too quiet
            return None
        
        # Pad or trim to ensure consistent length
        target_length = max(min_samples, len(audio_float))
        if len(audio_float) < target_length:
            # Pad with zeros
            audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
        elif len(audio_float) > self.sample_rate * 30:  # Limit to 30 seconds max
            audio_float = audio_float[-self.sample_rate * 30:]
        
        try:
            # Use no_speech_threshold to avoid transcribing silence
            result = self.audio_model.transcribe(
                audio_float, 
                fp16=torch.cuda.is_available(),
                language=language,
                no_speech_threshold=0.4,  # Higher threshold to filter silence better
                logprob_threshold=-1.0,   
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False  # Avoid context from previous transcriptions
            )
            text = result.get('text', '').strip()
            
            # Filter out very short, repetitive, or low-quality results
            if len(text) < 2 or text.lower() in ['you', 'the', 'a', 'i', 'to', 'and', 'of']:
                return None
                
            return text
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return None

    def add_transcription_line(self, text):
        with self.lock:
            if text is None or not text.strip():
                return
            
            # Rate limiting to prevent spam
            current_time = time.time()
            if current_time - self.last_transcription_time < self.min_transcription_interval:
                return
            
            # Avoid adding duplicate consecutive transcriptions
            if self.transcription and self.transcription[-1] == text:
                return
            
            # Filter out single words that are likely false positives
            if len(text.split()) == 1 and len(text) < 4:
                return
                
            self.transcription.append(text)
            self.last_transcription_time = current_time
            print(f"[Transcribed]: {text}")  # Debug output
            
            # Keep manageable history
            if len(self.transcription) > 20:
                self.transcription = self.transcription[-20:]

    def get_display_text(self, max_lines=5, max_chars=200):
        with self.lock:
            if not self.transcription:
                return ""
            
            lines = self.transcription[-max_lines:]
            display = '\n'.join(lines)
            
            if len(display) > max_chars:
                display = "..." + display[-(max_chars-3):]
            
            return display

# ---------------- Real-Time Visualization + Whisper ----------------
class RealTimePitchogramWhisper:
    def __init__(self, chunk_size=1024, sample_rate=16000, sai_width=400,
                 whisper_model="base", whisper_interval=2.0):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_interval = whisper_interval

        # CARFAC and pitchogram
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.pitchogram = RealTimePitchogram(num_channels=self.carfac.n_channels, sai_width=sai_width)
        self.n_channels = self.carfac.n_channels

        # Whisper
        self.whisper_handler = WhisperHandler(model_name=whisper_model)

        # Audio buffering for Whisper
        self.audio_queue = queue.Queue(maxsize=50)
        self.whisper_audio_buffer = []
        self.whisper_buffer_lock = threading.Lock()
        self.last_whisper_time = time.time()
        
        # Energy-based voice activity detection
        self.energy_threshold = 0.01
        self.silence_counter = 0
        self.max_silence_chunks = 10  # ~0.5 seconds of silence before processing

        # Visualization
        self.temporal_buffer_width = 200
        self.temporal_buffer = np.zeros((self.n_channels, self.temporal_buffer_width))
        self.audio_buffer = np.zeros(self.temporal_buffer_width)
        self._setup_visualization()

        # PyAudio
        self.p = None
        self.stream = None
        self.running = False

    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        cmap = self._create_enhanced_colormap()
        self.cmap = cmap
        
        self.im = self.ax.imshow(self.temporal_buffer, aspect='auto', origin='lower',
                                 cmap=cmap, interpolation='bilinear', vmin=0, vmax=1,
                                 extent=[0, self.temporal_buffer_width, 0, self.n_channels])
        
        self.pitch_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', horizontalalignment='right',
                                       fontsize=10, color='white', weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        self.transcription_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                              verticalalignment='bottom', fontsize=11,
                                              color='white', weight='bold',
                                              bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        # Waveform overlay
        x = np.arange(self.temporal_buffer_width)
        y = np.zeros_like(x)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        self.waveform_line = LineCollection(segments, linewidths=2, alpha=0.7)
        self.waveform_line.set_array(np.zeros(self.temporal_buffer_width - 1))
        self.ax.add_collection(self.waveform_line)
        
        plt.tight_layout()
        self.ax.set_title('Real-Time Audio Analysis with Whisper', fontsize=14, color='white', weight='bold')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('black')

    def _create_enhanced_colormap(self):
        colors = ['#000022', '#000055', '#0033AA', '#0066FF', '#00AAFF',
                  '#00FFAA', '#33FF77', '#77FF33', '#AAFF00', '#FFAA00',
                  '#FF7700', '#FF3300', '#FF0044', '#CC0077', '#FFFFFF']
        return LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            # Convert int16 to float32
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processing queue
            try:
                self.audio_queue.put_nowait(audio_float)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()  # Remove oldest
                    self.audio_queue.put_nowait(audio_float)
                except queue.Empty:
                    pass

            # Add to Whisper buffer
            with self.whisper_buffer_lock:
                self.whisper_audio_buffer.extend(audio_float)
                
                # Keep buffer manageable (max 10 seconds)
                max_buffer_size = self.sample_rate * 10
                if len(self.whisper_audio_buffer) > max_buffer_size:
                    excess = len(self.whisper_audio_buffer) - max_buffer_size
                    self.whisper_audio_buffer = self.whisper_audio_buffer[excess:]

        except Exception as e:
            print(f"Audio callback error: {e}")
        
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Process with CARFAC
                nap_output = self.carfac.process_chunk(audio_chunk)
                pitch_frame = self.pitchogram.run_frame(nap_output)

                # Update temporal buffer for visualization
                self.temporal_buffer[:, :-1] = self.temporal_buffer[:, 1:]
                self.temporal_buffer[:, -1] = pitch_frame.max(axis=1)

                # Update audio waveform buffer
                chunk_len = len(audio_chunk)
                if chunk_len <= self.temporal_buffer_width:
                    self.audio_buffer[:-chunk_len] = self.audio_buffer[chunk_len:]
                    self.audio_buffer[-chunk_len:] = audio_chunk
                else:
                    self.audio_buffer[:] = audio_chunk[-self.temporal_buffer_width:]

                # Voice activity detection for Whisper processing
                energy = np.mean(np.square(audio_chunk))
                if energy > self.energy_threshold:
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                continue

    def whisper_processing_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                # Process Whisper periodically or when silence is detected
                should_process = (
                    (current_time - self.last_whisper_time) >= self.whisper_interval and
                    self.silence_counter >= self.max_silence_chunks
                )
                
                if should_process:
                    with self.whisper_buffer_lock:
                        # Check if we have enough audio data
                        min_required_samples = int(self.sample_rate * 1.5)  # At least 1.5 seconds
                        
                        if len(self.whisper_audio_buffer) >= min_required_samples:
                            # Copy buffer for processing
                            audio_to_process = np.array(self.whisper_audio_buffer, dtype=np.float32)
                            
                            # Check audio quality before processing
                            energy = np.mean(np.square(audio_to_process))
                            if energy > self.energy_threshold * 0.5:  # Only process if there's some audio activity
                                
                                # Clear processed audio from buffer (keep last 0.5 seconds for context)
                                overlap_size = int(self.sample_rate * 0.5)
                                if len(self.whisper_audio_buffer) > overlap_size:
                                    self.whisper_audio_buffer = self.whisper_audio_buffer[-overlap_size:]
                                else:
                                    self.whisper_audio_buffer = []
                                
                                # Process in separate thread to avoid blocking
                                threading.Thread(
                                    target=self._process_whisper_chunk,
                                    args=(audio_to_process,),
                                    daemon=True
                                ).start()
                            else:
                                # Clear buffer if audio is too quiet
                                self.whisper_audio_buffer = []
                            
                    self.last_whisper_time = current_time
                    self.silence_counter = 0
                
                time.sleep(0.2)  # Slightly longer sleep to reduce CPU usage
                
            except Exception as e:
                print(f"Whisper processing loop error: {e}")
                time.sleep(0.5)

    def _process_whisper_chunk(self, audio_data):
        try:
            text = self.whisper_handler.transcribe_audio(audio_data, language='en')
            if text:
                self.whisper_handler.add_transcription_line(text)
        except Exception as e:
            print(f"Whisper chunk processing error: {e}")

    def _analyze_pitch_content(self):
        current_frame = self.temporal_buffer[:, -1]
        if np.max(current_frame) > 0.15:
            max_ch = np.argmax(current_frame)
            freq_ratio = max_ch / max(1, len(current_frame) - 1)
            estimated_freq = 80 * (8000 / 80) ** freq_ratio
            intensity = np.max(current_frame)
            return f"Pitch: ~{estimated_freq:.0f} Hz (Intensity: {intensity:.2f})"
        return "No clear pitch detected"

    def update_visualization(self, frame):
        try:
            # Update pitchogram
            current_max = np.max(self.temporal_buffer) if self.temporal_buffer.size else 0.001
            self.im.set_data(self.temporal_buffer)
            self.im.set_clim(vmin=0, vmax=max(0.001, min(1.0, current_max * 1.3)))
            
            # Update pitch analysis
            self.pitch_text.set_text(self._analyze_pitch_content())
            
            # Update transcription
            transcription_text = self.whisper_handler.get_display_text()
            self.transcription_text.set_text(transcription_text)
            
            # Update waveform
            waveform_scaled = self.n_channels * 0.1 + (self.n_channels * 0.3) * self.audio_buffer
            x = np.arange(self.temporal_buffer_width)
            y = waveform_scaled
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            self.waveform_line.set_segments(segments)
            colors = np.abs(self.audio_buffer[:-1])
            self.waveform_line.set_array(colors)
            self.waveform_line.set_cmap(self.cmap)
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im, self.pitch_text, self.transcription_text, self.waveform_line]

    def start(self):
        print("Starting real-time audio analysis with Whisper...")
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False
            )
            print(f"Audio stream opened: {self.sample_rate}Hz, {self.chunk_size} frames/buffer")
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            self.cleanup()
            return

        # Start processing threads
        self.running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
        threading.Thread(target=self.whisper_processing_loop, daemon=True).start()
        
        # Start audio stream
        self.stream.start_stream()
        print("System started. Speak into the microphone. Press Ctrl+C to stop.")
        
        # Start visualization with explicit save_count to avoid warning
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=50, blit=False,
            cache_frame_data=False  # Disable caching to avoid warning
        )
        plt.show()

    def cleanup(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
        try:
            if self.p:
                self.p.terminate()
        except:
            pass

    def stop(self):
        self.cleanup()
        plt.close('all')
        print("System stopped.")

# ---------------- Main ----------------
if __name__ == "__main__":
    system = None
    try:
        system = RealTimePitchogramWhisper(
            chunk_size=1024,          # Slightly larger chunks for better performance
            sample_rate=16000,        # Standard rate for Whisper
            sai_width=400,
            whisper_model="base",     # Start with base model (faster than medium)
            whisper_interval=2.0      # Process every 2 seconds
        )
        system.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if system:
            system.stop()
    except Exception as e:
        print(f"Error: {e}")
        if system:
            system.stop()
    finally:
        print("Cleanup complete.")