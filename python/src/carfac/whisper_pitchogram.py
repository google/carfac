import sys
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
from dataclasses import dataclass

import argostranslate.package
import argostranslate.translate

sys.path.append('./jax')
import jax
import jax.numpy as jnp
import carfac.jax.carfac as carfac

from carfac.np.carfac import CarParams
import sai

@dataclass
class SAIParams:
    """
    These are fake numbers. Use full constructor.
    """
    num_channels: int = 200
    sai_width: int = 400
    future_lags: int = 5
    num_triggers_per_frame: int = 10
    trigger_window_width: int = 20
    input_segment_width: int = 30
    channel_smoothing_scale: float = 0.5

    def __dict__(self):
        return {
            "num_channels": self.num_channels,
            "sai_width": self.sai_width,
            "future_lags": self.future_lags,
            "num_triggers_per_frame": self.num_triggers_per_frame,
            "trigger_window_width": self.trigger_window_width,
            "input_segment_width": self.input_segment_width,
            "channel_smoothing_scale": self.channel_smoothing_scale,
        }

@dataclass
class PitchogramParams:
    log_lag: bool = True
    lags_per_octave: float = 36.0
    min_lag_s: float = 0.0005
    log_offset_s: float = 0.0025
    vowel_time_constant_s: float = 0.02
    light_theme: bool = False

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
        return np.array(naps[:, :, 0]).T

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
    def __init__(self, model_name="base", non_english=False, debug=True):
        self.debug = debug
        model = model_name
        if model_name != "large" and not non_english:
            model = model + ".en"
        
        try:
            self.audio_model = whisper.load_model(model)
            self.sample_rate = 16000
        except Exception as e:
            try:
                self.audio_model = whisper.load_model("tiny")
                self.sample_rate = 16000
            except Exception as e2:
                self.audio_model = None
                self.sample_rate = 16000
        
        self.transcription = []
        self.lock = threading.Lock()
        self.last_transcription_time = time.time()
        self.min_transcription_interval = 0.1

    def transcribe_audio(self, audio_data, language='en'):
        
        # Reduced minimum length for more responsive transcription
        min_samples = int(self.sample_rate * 0.5)  # Reduced from 1.0 second
        
        # Ensure audio is float32 in range [-1, 1]
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_float).max()
        if max_val > 1.0:
            audio_float = audio_float / max_val
        
        # Pad or trim to ensure consistent length
        target_length = max(min_samples, len(audio_float))
        if len(audio_float) < target_length:
            audio_float = np.pad(audio_float, (0, target_length - len(audio_float)), 'constant')
        elif len(audio_float) > self.sample_rate * 30:
            audio_float = audio_float[-self.sample_rate * 30:]
        
        try:
            # More permissive Whisper settings
            result = self.audio_model.transcribe(
                audio_float, 
                fp16=torch.cuda.is_available(),
                language=language,
                #no_speech_threshold=0.6,  # Lowered from 0.4 - more sensitive to speech
                #logprob_threshold=-1.0,   # More permissive
                #compression_ratio_threshold=3.0,  # More permissive
                condition_on_previous_text=False
            )
            text = result.get('text', '').strip()
            
            # More lenient filtering
            if len(text) < 1:
                return None
                
            return text
        except Exception:
            return None

    def add_transcription_line(self, text):
        with self.lock:
            if text is None or not text.strip():
                return
            
            # More lenient rate limiting
            current_time = time.time()
            if current_time - self.last_transcription_time < self.min_transcription_interval:
                return
            
            # Avoid adding duplicate consecutive transcriptions
            if self.transcription and self.transcription[-1] == text:
                return
                
            self.transcription.append(text)
            self.last_transcription_time = current_time
            print(f"[Transcribed]: {text}")
            
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

# ---------------- Translation Handler ----------------
class TranslationHandler:
    """
    Handles the translation of transcribed audio.
    Exclusively uses argostranslate as the backend right now.
    """
    def __init__(self, from_lang="en", to_lang="zh"):
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.model = None
        self.loaded = self._load_model()

    def _load_model(self) -> bool: 
        # Load the translation model

        try:
            # Check the installed models first.
            installed_packages = argostranslate.package.get_installed_packages()
            is_cached = list(filter(lambda x: x.from_code == self.from_lang and x.to_code == self.to_lang, installed_packages))
            print(f"Translation model {self.from_lang} -> {self.to_lang} cached: {list(is_cached)}")

            if not is_cached:
                # Download and install the translation model if not found.
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()
                package_to_install = next(
                    filter(
                        lambda x: x.from_code == self.from_lang and x.to_code == self.to_lang, available_packages
                    )
                )
                argostranslate.package.install_from_path(package_to_install.download())
        except Exception as e:
            print(f"Error loading translation model: {e}")
            return False
        
        installed_languages = argostranslate.translate.get_installed_languages()
        from_model = next(filter(lambda x: x.code == self.from_lang, installed_languages), None)
        to_model = next(filter(lambda x: x.code == self.to_lang, installed_languages), None)

        self.model = from_model.get_translation(to_model)
        return True

    def translate(self, from_text: str) -> str:
        # Perform translation
        if not self.loaded:
            print("Translation model not loaded.")
            return ""

        to_text: str = self.model.translate(from_text)

        return to_text

# ---------------- Visualization Handler ----------------
class VisualizationHandler:
    @dataclass
    class ResamplingCell:
        left_index: int
        right_index: int
        left_weight: float
        interior_weight: float
        right_weight: float

        def __init__(self, left_edge: float, right_edge: float):
            cell_width: float = right_edge - left_edge

            if (cell_width < 1.0):
                grow: float = 0.5 * (1.0 - cell_width)
                left_edge -= grow
                right_edge += grow
            
            left_edge = max(0.0, left_edge)
            right_edge = max(0.0, right_edge)
            cell_width = right_edge - left_edge

            self.left_index = int(round(left_edge))
            self.right_index = int(round(right_edge))
            if (self.right_index > self.left_index and cell_width > 0.999):
                self.left_weight = (0.5 - (left_edge - self.left_index)) / cell_width
                self.interior_weight = 1.0 / cell_width
                self.right_weight = (0.5 + (right_edge - self.right_index)) / cell_width
            else:
                self.left_weight = 1.0
                self.interior_weight = 0.0
                self.right_weight = 0.0

        def CellAverage(self, samples: np.ndarray) -> float:
            if (self.left_index == self.right_index):
                return samples[self.left_index]
            return self.left_weight * samples[self.left_index] + \
                self.interior_weight * samples[self.left_index + 1 : self.right_index].sum() + \
                self.right_weight * samples[self.right_index]

    def __init__(self, sample_rate_hz: int, car_params: CarParams = CarParams(), sai_params: SAIParams = SAIParams(),
                 pitchogram_params: PitchogramParams = PitchogramParams()):
        self.temporal_buffer = np.zeros((sai_params.num_channels, sai_params.sai_width))

        # save the params, i guess
        self.car_params = car_params
        self.sai_params = sai_params
        self.pitchogram_params = pitchogram_params
        self.sample_rate_hz = sample_rate_hz

        # workspace, output declarations
        self.workspace = np.zeros((sai_params.sai_width))
        self.output = np.zeros((sai_params.sai_width))

        # get pole frequencies
        self.pole_frequencies = self.car_pole_frequencies(sample_rate_hz, car_params)

        # Create a binary mask to suppress the SAI's zero-lag peak. For the cth row,
        # we get the cth pole frequency and mask lags within half a cycle of zero.
        # (chris: I don't know what this means.)
        self.mask = np.ones((sai_params.num_channels, sai_params.sai_width), dtype=bool)
        center: int = sai_params.sai_width - sai_params.future_lags
        # pole_frequencies is a 1D array, so this is basically .size()
        for c in range(self.pole_frequencies.shape[0]):
            half_cycle_samples: float = 0.5 * sample_rate_hz / self.pole_frequencies[c]
            i_start: int = int(np.clip(np.floor(center - half_cycle_samples), 0, sai_params.sai_width - 1))
            i_end: int = int(np.clip(np.floor(center + half_cycle_samples), 0, sai_params.sai_width - 1))
            self.mask[c, i_start:i_end+1] = 0

        # create the vowel matrix
        self.vowel_matrix = self.create_vowel_matrix(sai_params.num_channels)
        self.vowel_coords = np.zeros((2, 1), dtype=np.float32)

        # set up the cgram
        frame_rate_hz: float = sample_rate_hz / sai_params.input_segment_width
        self.cgram_smoother = 1 - np.exp(-1 / (pitchogram_params.vowel_time_constant_s * frame_rate_hz))
        self.cgram = np.zeros(self.pole_frequencies.shape, dtype=np.float32)

        # log lag stuff
        self.log_lag_cells: list[VisualizationHandler.ResamplingCell] = list()
        if (not pitchogram_params.log_lag):
            self.output.resize((sai_params.sai_width))
        else:
            # If log_lag is true, set up ResamplingCells to warp the pitchogram to log axis.
            # The ith cell covers an interval...
            spacing: float = np.exp2(1.0 / pitchogram_params.lags_per_octave)
            log_offset: float = sample_rate_hz * pitchogram_params.log_offset_s
            left_edge: float = sample_rate_hz * pitchogram_params.min_lag_s

            while True:
                right_edge: float = (left_edge + log_offset) * spacing - log_offset
                cell: VisualizationHandler.ResamplingCell = self.ResamplingCell(left_edge, right_edge)
                if (cell.right_index >= sai_params.sai_width):
                    break
                self.log_lag_cells.append(cell)
                left_edge = right_edge

            self.workspace.resize((sai_params.sai_width))
            self.output.resize((len(self.log_lag_cells)))

        self.img = np.zeros((sai_params.sai_width, self.output.shape[0], 3), dtype=np.uint8)
        # print("VisualizationHandler shapes: ", self.workspace.shape, self.output.shape, self.mask.shape)

    def car_pole_frequencies(self, sample_rate_hz, car_params: CarParams) -> np.ndarray:
            """
            Ported from car.cc: Returns array of pole frequencies for CAR channels.
            car_params: dict with keys matching C++ CARParams
            sample_rate_hz: float
            Returns: np.ndarray of pole frequencies
            """
            num_channels: int = 0
            pole_hz: float = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
            while pole_hz > car_params.min_pole_hz:
                num_channels += 1
                pole_hz -= car_params.erb_per_step * \
                    ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)

            pole_freqs = np.zeros(num_channels, dtype=np.float32)
            pole_hz = car_params.first_pole_theta * sample_rate_hz / (2.0 * np.pi)
            for channel in range(num_channels):
                pole_freqs[channel] = pole_hz
                pole_hz -= car_params.erb_per_step * \
                    ((car_params.erb_break_freq + pole_hz) / car_params.erb_q)
            return pole_freqs
    
    def create_vowel_matrix(self, num_channels, erb_per_step=0.5) -> np.ndarray:
        """
        Ported from pitchogram.cc: creates a vowel embedding matrix for F1 and F2 formants.
        Returns: vowel_matrix (2, num_channels)
        """
        def kernel(center, c):
            z = (c - center) / 3.3
            return np.exp((z * z) / -2)

        # These frequency values are from the C++ code
        f2_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 2365)
        f2_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 1100)
        f1_hi = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 700)
        f1_lo = self.frequency_to_channel_index(self.sample_rate_hz, erb_per_step, 265)

        vowel_matrix = np.zeros((2, num_channels), dtype=np.float32)
        for c in range(num_channels):
            vowel_matrix[0, c] = kernel(f2_lo, c) - kernel(f2_hi, c)
            vowel_matrix[1, c] = kernel(f1_lo, c) - kernel(f1_hi, c)
        vowel_matrix *= erb_per_step / 2
        return vowel_matrix

    # TODO: probably extract this
    def frequency_to_channel_index(self, sample_rate_hz: int, erb_per_step: float, pole_freq: int):
        """
        Implements CARFrequencyToChannelIndex from car.cc.
        freq: frequency to convert
        erb_per_step: ERB spacing in Hz
        num_channels: number of channels
        Returns: channel index (float)
        """
        first_pole_theta: float = 0.85 * np.pi # default value from CARParams
        erb_q: float = 1000 / (24.7 * 4.37)

        pole0_hz: float = first_pole_theta * sample_rate_hz / (2.0 * np.pi)
        break_freq: float = 165.3  # The Greenwood map's break frequency in Hertz.
        ratio: float = 1 - erb_per_step / erb_q
        min_pole_hz: float = 30 # default value from CARParams
        # Clamp freq
        pole_freq = np.clip(pole_freq, min_pole_hz, pole0_hz)
        top = np.log((pole_freq + break_freq) / (pole0_hz + break_freq))
        bottom = np.log(ratio)
        return top / bottom

    def get_vowel_embedding(self, nap) -> np.ndarray:
        """
        Ported from pitchogram.cc: computes vowel embedding coordinates.
        nap: (num_channels, ...)
        Returns: vowel_coords (2,)
        """
        self.cgram += self.cgram_smoother * (nap.mean(axis=1) - self.cgram)
        self.vowel_coords = self.vowel_matrix @ self.cgram
        return self.vowel_coords

    def run_frame(self, sai_frame: np.ndarray) -> np.ndarray:
        # Process the SAI frame
        if (not self.pitchogram_params.log_lag):
            self.output = (sai_frame * self.mask).mean(axis=0)
        else:
            self.workspace = (sai_frame * self.mask).mean(axis=0)
            for i in range(self.output.shape[0]):
                self.output[i] = self.log_lag_cells[i].CellAverage(self.workspace)

        return self.output

    def draw_column(self, column_ptr: np.ndarray) -> None:
        # Update the visualization frame
        # Ensure self.vowel_coords is a flat array for scalar indexing
        v = np.ravel(self.vowel_coords)
        tint = np.array([
            0.5 - 0.6 * v[1],
            0.5 - 0.6 * v[0],
            0.35 * (v[0] + v[1]) + 0.4
        ], dtype=np.float32)

        k_scale: float = 0.5 * 255
        tint *= k_scale

        # print("draw_column:", column_ptr.shape, self.output.shape)

        for i in range(self.output.shape[0]):
            column_ptr[i] = np.clip(np.int32((tint * self.output[i])), 0, 255)

        pass

# ---------------- Real-Time Visualization + Whisper ----------------
class RealTimePitchogramWhisper:
    def __init__(self, chunk_size=1024, sample_rate=16000, sai_width=200,
                 whisper_model="tiny", whisper_interval=1.5, 
                 enable_translation=False, from_lang="zh", to_lang="en",
                 debug=True):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.sai_width = sai_width
        self.whisper_interval = whisper_interval
        self.debug = debug

        # CARFAC, SAI and pitchogram
        self.carfac = RealCARFACProcessor(fs=sample_rate)
        self.pitchogram = RealTimePitchogram(num_channels=self.carfac.n_channels, sai_width=sai_width)
        self.n_channels = self.carfac.n_channels

        # SAI parameters
        self.sai_params = SAIParams(
            num_channels=self.n_channels,
            sai_width=self.sai_width,
            future_lags=self.sai_width - 1,
            num_triggers_per_frame=2, # PPP.num_triggers_per_frame
            trigger_window_width=self.chunk_size + 1,   # input_segment_width + 1
            input_segment_width=self.chunk_size,  # num_samples_per_segment
            channel_smoothing_scale=0.5
        )
        self.SAI = sai.SAI(self.sai_params)

        # Whisper
        self.whisper_handler = WhisperHandler(model_name=whisper_model, debug=debug)

        # Translation
        self.translation_handler = None
        if enable_translation:
            self.translation_handler = TranslationHandler(from_lang=from_lang, to_lang=to_lang)

        # Audio buffering for Whisper
        self.audio_queue = queue.Queue(maxsize=50)
        self.whisper_audio_buffer = []
        self.whisper_buffer_lock = threading.Lock()
        self.last_whisper_time = time.time()
        
        # Much more sensitive voice activity detection
        self.energy_threshold = 0.0001  # Significantly lowered
        self.silence_counter = 0
        self.max_silence_chunks = 5  # Reduced silence requirement
        self.audio_chunk_counter = 0

        # Visualization
        self.visualization_handler = VisualizationHandler(sample_rate, sai_params=self.sai_params)

        self.temporal_buffer_width = 200
        self.temporal_buffer = np.zeros((self.n_channels, self.temporal_buffer_width))
        self.audio_buffer = np.zeros(self.temporal_buffer_width)
        self._setup_visualization()

        # PyAudio
        self.p = None
        self.stream = None
        self.running = False

    def _setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 16))
        cmap = self._create_enhanced_colormap()
        self.cmap = cmap
        
        self.im = self.ax.imshow(self.visualization_handler.img, aspect='auto', origin='lower',
                                 interpolation='bilinear',
                                 extent=[0, self.n_channels, 0, self.temporal_buffer_width])

        self.pitch_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                       verticalalignment='top', horizontalalignment='right',
                                       fontsize=10, color='white', weight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        self.transcription_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                              verticalalignment='bottom', fontsize=12,
                                              color='lime', weight='bold',
                                              bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # Debug info text
        self.debug_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', fontsize=8,
                                      color='yellow', weight='normal',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        self.ax.axis('off')

    def _create_enhanced_colormap(self):
        colors = ['#000022', '#000055', '#0033AA', '#0066FF', '#00AAFF',
                  '#00FFAA', '#33FF77', '#77FF33', '#AAFF00', '#FFAA00',
                  '#FF7700', '#FF3300', '#FF0044', '#CC0077', '#FFFFFF']
        return LinearSegmentedColormap.from_list("enhanced_audio", colors, N=256)

    def list_audio_devices(self):
        """List available audio input devices"""
        if self.p is None:
            self.p = pyaudio.PyAudio()
        
        print("\nAvailable audio input devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  Device {i}: {info['name']} (Channels: {info['maxInputChannels']})")
        print()

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            # Convert int16 to float32
            audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            energy = np.mean(np.square(audio_float))
            
            self.audio_chunk_counter += 1

            
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
                
                # Keep buffer manageable (max 15 seconds for better context)
                max_buffer_size = self.sample_rate * 15
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

                # Process with SAI
                sai_output = self.SAI.RunSegment(nap_output)
                self.visualization_handler.get_vowel_embedding(nap_output)
                pitch_frame = self.visualization_handler.run_frame(sai_output)

                # Update temporal buffer for visualization
                self.visualization_handler.img[:-1, :] = self.visualization_handler.img[1:, :]
                self.visualization_handler.draw_column(self.visualization_handler.img[-1, :])

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
                raise e
                continue

    def whisper_processing_loop(self):
        while self.running:
            try:
                current_time = time.time()
                
                with self.whisper_buffer_lock:
                    buffer_size = len(self.whisper_audio_buffer)
                    if buffer_size > 0:
                        audio_energy = np.mean(np.square(self.whisper_audio_buffer[-min(1000, buffer_size):]))
                    else:
                        audio_energy = 0
                
                # Simplified processing logic - process based on time interval OR sufficient audio
                time_condition = (current_time - self.last_whisper_time) >= self.whisper_interval
                buffer_condition = buffer_size >= int(self.sample_rate * 1.0)  # At least 1 second of audio
                
                should_process = time_condition and buffer_condition
                
                if self.debug and self.audio_chunk_counter % 100 == 0:
                    print(f"DEBUG: Buffer size: {buffer_size/self.sample_rate:.1f}s, "
                          f"Energy: {audio_energy:.6f}, "
                          f"Should process: {should_process}, "
                          f"Time since last: {current_time - self.last_whisper_time:.1f}s")
                
                if should_process:
                    with self.whisper_buffer_lock:
                        min_required_samples = int(self.sample_rate * 0.8)  # Reduced minimum
                        
                        if len(self.whisper_audio_buffer) >= min_required_samples:
                            # Copy buffer for processing
                            audio_to_process = np.array(self.whisper_audio_buffer, dtype=np.float32)
                            
                            # Check audio quality before processing
                            energy = np.mean(np.square(audio_to_process))
                            max_amplitude = np.max(np.abs(audio_to_process))
                            
                            if self.debug:
                                print(f"DEBUG: Processing audio chunk: {len(audio_to_process)/self.sample_rate:.1f}s, "
                                      f"energy={energy:.6f}, max_amp={max_amplitude:.4f}")
                            
                            # More lenient energy check
                            if energy > self.energy_threshold * 0.1 and max_amplitude > 0.001:
                                
                                # Keep more overlap for better context
                                overlap_size = int(self.sample_rate * 1.0)  # 1 second overlap
                                if len(self.whisper_audio_buffer) > overlap_size:
                                    self.whisper_audio_buffer = self.whisper_audio_buffer[-overlap_size:]
                                else:
                                    self.whisper_audio_buffer = []
                                
                                # Process in separate thread
                                threading.Thread(
                                    target=self._process_whisper_chunk,
                                    args=(audio_to_process,),
                                    daemon=True
                                ).start()
                            else:
                                if self.debug:
                                    print(f"DEBUG: Skipping processing - insufficient audio quality")
                                # Only clear buffer if audio is extremely quiet
                                if energy < self.energy_threshold * 0.01:
                                    self.whisper_audio_buffer = []
                            
                    self.last_whisper_time = current_time
                
                time.sleep(0.1)  # Faster polling for more responsive transcription
                
            except Exception as e:
                print(f"Whisper processing loop error: {e}")
                time.sleep(0.5)

    def _process_whisper_chunk(self, audio_data):
        try:
            if self.debug:
                print(f"DEBUG: Starting Whisper transcription for {len(audio_data)/self.sample_rate:.1f}s of audio")

            language = 'en' # Default language, uses from_lang if translation enabled
            if self.translation_handler:
                language = self.translation_handler.from_lang

            text = self.whisper_handler.transcribe_audio(audio_data, language=language)
            if text and len(text.strip()) > 0:
                self.whisper_handler.add_transcription_line(text)
                if self.debug:
                    print(f"DEBUG: Successfully transcribed: '{text}'")

                if self.translation_handler:
                    translated_text = self.translation_handler.translate(text)
                    if translated_text:
                        print(f"{self.translation_handler.from_lang}-{self.translation_handler.to_lang}: {translated_text}")
            else:
                if self.debug:
                    print(f"DEBUG: No transcription result")
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
            self.im.set_data(self.visualization_handler.img)
            self.im.set_clim(vmin=0, vmax=max(0.001, min(1.0, current_max * 1.3)))
            
            # Update pitch analysis
            self.pitch_text.set_text(self._analyze_pitch_content())
            
            # Update transcription
            transcription_text = self.whisper_handler.get_display_text()
            if not transcription_text:
                transcription_text = "Listening... (speak into microphone)"
            self.transcription_text.set_text(transcription_text)

            # Update debug info
            if self.debug:
                with self.whisper_buffer_lock:
                    buffer_seconds = len(self.whisper_audio_buffer) / self.sample_rate
                current_energy = np.mean(np.square(self.audio_buffer)) if len(self.audio_buffer) > 0 else 0
                debug_info = f"Buffer: {buffer_seconds:.1f}s | Energy: {current_energy:.4f} | Threshold: {self.energy_threshold:.4f}"
                self.debug_text.set_text(debug_info)
            
            
        except Exception as e:
            print(f"Visualization update error: {e}")
        
        return [self.im, self.pitch_text, self.transcription_text, self.debug_text]

    def start(self):
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # List available devices for debugging
        if self.debug:
            self.list_audio_devices()
        
        try:
            # Try to use default input device first
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
            print(f"Failed to open audio stream with default device: {e}")
            # Try to find a working input device
            for i in range(self.p.get_device_count()):
                try:
                    info = self.p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"Trying device {i}: {info['name']}")
                        self.stream = self.p.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=self.sample_rate,
                            input=True,
                            input_device_index=i,
                            frames_per_buffer=self.chunk_size,
                            stream_callback=self.audio_callback,
                            start=False
                        )
                        print(f"Successfully opened device {i}")
                        break
                except:
                    continue
            else:
                print("Failed to open any audio input device")
                self.cleanup()
                return

        # Start processing threads
        self.running = True
        threading.Thread(target=self.process_audio, daemon=True).start()
        threading.Thread(target=self.whisper_processing_loop, daemon=True).start()
        
        # Start audio stream
        self.stream.start_stream()
        print("System started. Speak into the microphone. Press Ctrl+C to stop.")
        print("Debug mode enabled - watch the console for detailed information.")
        
        # Start visualization
        self.animation = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=50, blit=False,
            cache_frame_data=False
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
            chunk_size=512,
            sample_rate=16000,
            sai_width=400,
            whisper_model="middle",    # Using tiny model for faster processing
            whisper_interval=1.5,    # Process every 1.5 seconds
            enable_translation=False,
            from_lang="zh",
            to_lang="en",
            debug=True               # Enable debug mode
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
        raise e
    finally:
        print("Cleanup complete.")