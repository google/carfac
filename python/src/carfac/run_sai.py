import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio
import threading
import queue
import time
import carfac.sai as pysai
from scipy.signal import butter, sosfilt, hilbert
from collections import deque

def CreateSAIParams(sai_width, num_triggers_per_frame=2, **kwargs):
    """Exactly like in sai_test.py - fills SAIParams with reasonable defaults"""
    return pysai.SAIParams(
        sai_width=sai_width,
        future_lags=sai_width // 2,  # Half from future
        num_triggers_per_frame=num_triggers_per_frame,
        **kwargs,
    )

class WorkingCARFACPitchogram:
    def __init__(self):
        # Audio settings
        self.sample_rate = 22050
        self.chunk_size = 1024
        
        # SAI settings - using pattern from test
        self.input_segment_width = 882  # EXACTLY like the test
        self.num_channels = 71          # EXACTLY like the test
        self.sai_width = 100            # Good for pitch detection
        
        # Display settings - FANCY low pitch range like professional tools
        self.time_window = 15  # Longer time window for more data
        self.pitch_range = (40, 100)  # LOW PITCH RANGE - more detailed for bass/fundamental frequencies
        
        # History storage - MUCH larger for dense visualization
        self.pitch_history = deque(maxlen=2000)  # Store many more points
        self.time_history = deque(maxlen=2000)
        self.confidence_history = deque(maxlen=2000)
        
        # Audio buffer
        self.audio_queue = queue.Queue(maxsize=100)
        self.audio_buffer = np.zeros(self.sample_rate * 3)  # 3 second buffer
        self.buffer_index = 0
        
        # Current values
        self.current_pitch = 0
        self.current_confidence = 0
        self.current_time = 0
        
        # Setup CARFAC exactly like the test
        self.setup_carfac_from_test()
        
        # Audio
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def setup_carfac_from_test(self):
        """Setup CARFAC using the EXACT same pattern as sai_test.py"""
        print("Setting up CARFAC using sai_test.py pattern...")
        
        # Use the EXACT same function and parameters as the test
        self.sai_params = CreateSAIParams(
            num_channels=self.num_channels,
            input_segment_width=self.input_segment_width,
            trigger_window_width=self.input_segment_width,  # EXACTLY like test
            sai_width=self.sai_width,
        )
        
        try:
            # Create SAI exactly like the test
            self.sai = pysai.SAI(self.sai_params)
            
            # Test it like the test does
            self.test_sai()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def test_sai(self):
        
        try:
            # Create test pulse train like in the test
            test_segment = self.create_pulse_train(self.num_channels, self.input_segment_width, period=10)
            
            print(f"Test segment shape: {test_segment.shape}")
            print(f"Expected shape: ({self.num_channels}, {self.input_segment_width})")
            
                
        except Exception as e:
            print(f"✗ SAI test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_pulse_train(self, num_channels, num_samples, period, leading_zeros=0):
        """Create pulse train exactly like sai_test.py"""
        segment = np.zeros((num_channels, num_samples))
        for i in range(num_channels):
            # Begin each channel at a different phase
            phase = (i + leading_zeros) % period
            for j in range(phase, num_samples, period):
                segment[i, j] = 1
        return segment.astype(np.float32)
    
    def create_cochlear_channels(self, audio_segment):
        """Convert mono audio to multi-channel input for SAI"""
        # Need to create exactly input_segment_width samples
        if len(audio_segment) != self.input_segment_width:
            # Resample to exact size
            if len(audio_segment) > self.input_segment_width:
                # Downsample
                audio_segment = audio_segment[:self.input_segment_width]
            else:
                # Pad with zeros
                audio_segment = np.pad(audio_segment, (0, self.input_segment_width - len(audio_segment)))
        
        # Create multi-channel representation
        channels = []
        
        # Simple filterbank approach
        low_freq = 80
        high_freq = min(8000, self.sample_rate // 2 - 100)
        
        # Create frequency channels
        center_freqs = np.logspace(np.log10(low_freq), np.log10(high_freq), self.num_channels)
        
        for fc in center_freqs:
            try:
                # Bandpass filter
                bandwidth = fc * 0.25
                low_cut = max(fc - bandwidth/2, 20)
                high_cut = min(fc + bandwidth/2, self.sample_rate/2 - 50)
                
                if low_cut < high_cut:
                    sos = butter(2, [low_cut, high_cut], btype='band', 
                               fs=self.sample_rate, output='sos')
                    filtered = sosfilt(sos, audio_segment)
                    
                    # Simple envelope detection
                    envelope = np.abs(hilbert(filtered))
                    channels.append(envelope)
                else:
                    channels.append(np.zeros_like(audio_segment))
            except:
                channels.append(np.zeros_like(audio_segment))
        
        result = np.array(channels, dtype=np.float32)
        return result
    
    def extract_pitch_from_sai(self, sai_frame):
        """Extract MULTIPLE pitches from SAI frame for dense visualization"""
        try:
            if sai_frame is None or sai_frame.size == 0:
                return [], []
            
            # Sum across frequency channels
            if len(sai_frame.shape) == 2:
                summary = np.mean(sai_frame, axis=0)
            else:
                summary = sai_frame
            
            if len(summary) < 5:
                return [], []
            
            max_sai = np.max(summary)
            if max_sai < 0.001:
                return [], []
            
            # EXTRACT MULTIPLE PITCHES for dense visualization
            pitches_found = []
            confidences_found = []
            
            # Much more aggressive pitch extraction - find MANY peaks
            for threshold in [0.6, 0.4, 0.2, 0.1, 0.05, 0.02]:  # Multiple threshold levels
                peak_threshold = max_sai * threshold
                
                for i in range(1, len(summary)-1):
                    if (summary[i] > peak_threshold and 
                        summary[i] >= summary[i-1] and 
                        summary[i] >= summary[i+1]):
                        
                        # Convert bin to frequency
                        max_delay_ms = 25
                        delay_per_bin_ms = max_delay_ms / len(summary)
                        delay_ms = i * delay_per_bin_ms
                        
                        if delay_ms > 0.5:  # Valid delay
                            freq = 1000.0 / delay_ms
                            confidence = summary[i] / max_sai
                            
                            # Accept VERY wide frequency range, then filter for display
                            if 20 <= freq <= 200:  # Wide detection range
                                # Avoid duplicate nearby frequencies
                                is_duplicate = False
                                for existing_freq in pitches_found:
                                    if abs(freq - existing_freq) < 2:  # Within 2 Hz for precision
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    pitches_found.append(freq)
                                    confidences_found.append(confidence)
            
            return pitches_found, confidences_found
            
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return [], []
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def update_buffer(self, new_data):
        """Update circular buffer"""
        for sample in new_data:
            self.audio_buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % len(self.audio_buffer)
    
    def processing_thread(self):
        """Main processing thread"""
        print("Processing thread started")
        start_time = time.time()
        
        while self.is_running:
            try:
                # Get audio data
                audio_chunks = []
                for _ in range(5):  # Get several chunks
                    try:
                        chunk = self.audio_queue.get_nowait()
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if audio_chunks:
                    combined_audio = np.concatenate(audio_chunks)
                    self.update_buffer(combined_audio)
                    
                    # Extract exactly input_segment_width samples
                    start_idx = (self.buffer_index - self.input_segment_width) % len(self.audio_buffer)
                    
                    if start_idx + self.input_segment_width <= len(self.audio_buffer):
                        audio_segment = self.audio_buffer[start_idx:start_idx + self.input_segment_width]
                    else:
                        # Handle wraparound
                        part1 = self.audio_buffer[start_idx:]
                        part2 = self.audio_buffer[:self.input_segment_width - len(part1)]
                        audio_segment = np.concatenate([part1, part2])
                    
                    # Normalize audio and check levels
                    max_val = np.max(np.abs(audio_segment))
                    if max_val > 0:
                        audio_segment = audio_segment / max_val
                        # DEBUG: Check if we're getting audio
                        if max_val > 0.01:  # Only print for significant audio
                            print(f"Audio level: {max_val:.4f}")
                    else:
                        continue  # Skip silent segments
                    
                    try:
                        # Create multi-channel input
                        cochlear_input = self.create_cochlear_channels(audio_segment)
                        
                        # Verify exact shape
                        if cochlear_input.shape != (self.num_channels, self.input_segment_width):
                            print(f"Shape error: {cochlear_input.shape} != ({self.num_channels}, {self.input_segment_width})")
                            continue
                        
                        # Run SAI (exactly like the test)
                        sai_frame = self.sai.RunSegment(cochlear_input)
                        
                        if sai_frame is not None:
                            # Extract MULTIPLE pitches for dense visualization
                            pitches_found, confidences_found = self.extract_pitch_from_sai(sai_frame)
                            
                            # Add ALL found pitches to history
                            current_time = time.time() - start_time
                            
                            if pitches_found:  # If any pitches found
                                for pitch, confidence in zip(pitches_found, confidences_found):
                                    self.pitch_history.append(pitch)
                                    self.confidence_history.append(confidence)
                                    self.time_history.append(current_time)
                                
                                # Update current values with the strongest pitch
                                strongest_idx = np.argmax(confidences_found)
                                self.current_pitch = pitches_found[strongest_idx]
                                self.current_confidence = confidences_found[strongest_idx]
                                self.current_time = current_time
                            else:
                                # No pitch found - add zero entry to maintain timing
                                self.pitch_history.append(0)
                                self.confidence_history.append(0)
                                self.time_history.append(current_time)
                                self.current_pitch = 0
                                self.current_confidence = 0
                                self.current_time = current_time
                        
                    except Exception as e:
                        print(f"SAI processing error: {e}")
                        continue
                
                time.sleep(0.01)  # 100 FPS for very dense data
                
            except Exception as e:
                print(f"Processing thread error: {e}")
                time.sleep(0.1)
    
    def start_audio(self):
        """Start audio recording"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread_handle = threading.Thread(target=self.processing_thread, daemon=True)
            self.processing_thread_handle.start()
            
            print(f"✓ Audio started: {self.sample_rate} Hz")
            return True
            
        except Exception as e:
            print(f"Audio start error: {e}")
            return False
    
    def stop_audio(self):
        """Stop audio"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("Audio stopped")
    
    def freq_to_note(self, freq):
        """Convert frequency to musical note"""
        if freq <= 0:
            return "---"
        
        A4 = 440
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        semitones = 12 * np.log2(freq / A4)
        note_num = int(round(semitones)) + 9
        octave = 4 + (note_num // 12)
        note_idx = note_num % 12
        return f"{note_names[note_idx]}{octave}"
    
    def update_plot(self, frame):
        """Update the FANCY pitchogram display"""
        try:
            if len(self.pitch_history) < 2:
                return []
            
            # FANCY: Clear with style
            self.ax.clear()
            
            # Create ULTRA-DENSE visualization with FANCY styling
            times = np.array(self.time_history)
            pitches = np.array(self.pitch_history) 
            confidences = np.array(self.confidence_history)
            
            # Filter for display range (40-100 Hz)
            display_mask = (pitches >= self.pitch_range[0]) & (pitches <= self.pitch_range[1])
            display_times = times[display_mask]
            display_pitches = pitches[display_mask]
            display_confidences = confidences[display_mask]
            
            # BEAUTIFUL PAINTING-LIKE VISUALIZATION - NO DOTS, PURE COLOR STROKES
            if len(display_times) > 0:
                # Layer 1: Ultra-high confidence (BRIGHT COLOR STROKES)
                ultra_conf_mask = display_confidences > 0.7
                if np.any(ultra_conf_mask):
                    for size in [80, 60, 40, 20]:  # Multiple stroke sizes
                        self.ax.scatter(display_times[ultra_conf_mask], display_pitches[ultra_conf_mask],
                                      c=display_confidences[ultra_conf_mask], cmap='plasma',
                                      s=size, alpha=0.8, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')  # Square markers for paint effect
                
                # Layer 2: High confidence (BRIGHT PAINT STROKES)
                high_conf_mask = (display_confidences > 0.4) & (display_confidences <= 0.7)
                if np.any(high_conf_mask):
                    for size in [60, 40, 25]:
                        self.ax.scatter(display_times[high_conf_mask], display_pitches[high_conf_mask],
                                      c=display_confidences[high_conf_mask], cmap='hot',
                                      s=size, alpha=0.7, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')
                
                # Layer 3: Medium confidence (COLORFUL PAINT PATCHES)
                med_conf_mask = (display_confidences > 0.2) & (display_confidences <= 0.4)
                if np.any(med_conf_mask):
                    for size in [45, 30, 15]:
                        self.ax.scatter(display_times[med_conf_mask], display_pitches[med_conf_mask],
                                      c=display_confidences[med_conf_mask], cmap='rainbow',
                                      s=size, alpha=0.6, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')
                
                # Layer 4: Low confidence (SOFT COLOR WASHES)
                low_conf_mask = (display_confidences > 0.1) & (display_confidences <= 0.2)
                if np.any(low_conf_mask):
                    for size in [35, 20]:
                        self.ax.scatter(display_times[low_conf_mask], display_pitches[low_conf_mask],
                                      c=display_confidences[low_conf_mask], cmap='viridis',
                                      s=size, alpha=0.5, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')
                
                # Layer 5: Ultra-low confidence (SUBTLE COLOR TEXTURES)
                ultra_low_mask = (display_confidences > 0.05) & (display_confidences <= 0.1)
                if np.any(ultra_low_mask):
                    for size in [25, 15]:
                        self.ax.scatter(display_times[ultra_low_mask], display_pitches[ultra_low_mask],
                                      c=display_confidences[ultra_low_mask], cmap='cool',
                                      s=size, alpha=0.4, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')
                
                # Layer 6: Background texture (VERY SOFT PAINT WASHES)
                bg_mask = (display_confidences > 0.01) & (display_confidences <= 0.05)
                if np.any(bg_mask):
                    for size in [20, 10]:
                        self.ax.scatter(display_times[bg_mask], display_pitches[bg_mask],
                                      c=display_confidences[bg_mask], cmap='spring',
                                      s=size, alpha=0.3, vmin=0, vmax=1, 
                                      edgecolors='none', marker='s')
            
            # BEAUTIFUL PAINTING: Enhanced visual styling
            self.ax.set_xlim(max(0, self.current_time - self.time_window), self.current_time + 1)
            self.ax.set_ylim(self.pitch_range[0], self.pitch_range[1])
            
            # NO GRID - pure painting effect
            self.ax.grid(False)
            self.ax.set_facecolor('black')  # Pure black canvas
            
            # HIDE ALL LABELS AND TICKS for pure art
            self.ax.set_xlabel('')
            self.ax.set_ylabel('') 
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            
            # HIDE SPINES for borderless effect
            for spine in self.ax.spines.values():
                spine.set_visible(False)
            
            # PURE ART: Minimal status display
            if self.current_pitch > 0 and self.pitch_range[0] <= self.current_pitch <= self.pitch_range[1]:
                note = self.freq_to_note(self.current_pitch)
                title = f'{self.current_pitch:.1f} Hz ({note})'
                self.ax.set_title(title, fontsize=10, color='white', fontweight='light', pad=10)
            else:
                self.ax.set_title('', fontsize=1)  # Empty title
            
            return []
            
        except Exception as e:
            print(f"Plot update error: {e}")
            return []
    
    def run(self):
        """Run the real-time pitchogram"""
        print("=== ● ♫ ○ ◊ ===")
        print("♪ ▲ sai_test.py ♬")
        
        if not self.start_audio():
            print("✗ ✗ ○ ●!")
            return
        
        try:
            # PURE ART: Minimal setup
            plt.style.use('dark_background')
            self.fig, self.ax = plt.subplots(figsize=(16, 10))
            self.fig.patch.set_facecolor('black')
            
            # NO COLORBAR - pure art
            
            # MINIMAL TITLE
            self.fig.suptitle("", fontsize=1)
            
            # NO VERSION INFO - pure art
            
            # SMOOTH ANIMATION for painting effect
            self.anim = FuncAnimation(self.fig, self.update_plot, 
                                    interval=30,  # 33 FPS for smooth painting
                                    blit=False, cache_frame_data=False)
            
            print("")  # Silent start
            plt.show()
            
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Display error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_audio()
            self.audio.terminate()

def main():
    """Main function"""
    print("♫ ○ ◊ - ♪ ● sai_test.py")
    print("=" * 60)
    
    try:
        pitchogram = WorkingCARFACPitchogram()
        pitchogram.run()
        
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()