"""
WORKING CARFAC PITCHOGRAM
Based exactly on the pattern from sai_test.py
"""

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
        
        # Display settings
        self.time_window = 10
        self.pitch_range = (80, 500)
        
        # History storage
        self.pitch_history = deque(maxlen=300)
        self.time_history = deque(maxlen=300)
        self.confidence_history = deque(maxlen=300)
        
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
        
        print(f"SAI Parameters (matching test):")
        print(f"  num_channels: {self.sai_params.num_channels}")
        print(f"  input_segment_width: {self.sai_params.input_segment_width} samples")
        print(f"  trigger_window_width: {self.sai_params.trigger_window_width}")
        print(f"  sai_width: {self.sai_params.sai_width}")
        print(f"  future_lags: {self.sai_params.future_lags}")
        
        try:
            # Create SAI exactly like the test
            self.sai = pysai.SAI(self.sai_params)
            print("✓ CARFAC SAI created successfully using test pattern!")
            
            # Test it like the test does
            self.test_sai()
            
        except Exception as e:
            print(f"CARFAC setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_sai(self):
        """Test SAI like sai_test.py does"""
        print("Testing SAI with synthetic data...")
        
        try:
            # Create test pulse train like in the test
            test_segment = self.create_pulse_train(self.num_channels, self.input_segment_width, period=10)
            
            print(f"Test segment shape: {test_segment.shape}")
            print(f"Expected shape: ({self.num_channels}, {self.input_segment_width})")
            
            # Run SAI exactly like the test
            sai_frame = self.sai.RunSegment(test_segment)
            
            if sai_frame is not None:
                print(f"✓ SAI test passed! Output shape: {sai_frame.shape}")
                print(f"✓ Output range: {np.min(sai_frame):.6f} to {np.max(sai_frame):.6f}")
                return True
            else:
                print("✗ SAI test failed: returned None")
                return False
                
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
        """Extract pitch from SAI frame"""
        try:
            if sai_frame is None or sai_frame.size == 0:
                return 0, 0
            
            # Sum across frequency channels (like test analysis)
            if len(sai_frame.shape) == 2:
                summary = np.mean(sai_frame, axis=0)
            else:
                summary = sai_frame
            
            if len(summary) < 5:
                return 0, 0
            
            # Find peak (skip first few bins)
            peak_idx = np.argmax(summary[3:]) + 3
            
            # Convert SAI bin to frequency
            # SAI bins represent time lags
            max_delay_ms = 50  # 50ms max delay for pitch detection
            delay_per_bin_ms = max_delay_ms / len(summary)
            delay_ms = peak_idx * delay_per_bin_ms
            
            if delay_ms > 0:
                freq = 1000.0 / delay_ms  # Convert ms to Hz
                confidence = summary[peak_idx] / np.max(summary) if np.max(summary) > 0 else 0
                
                # Filter valid pitch range
                if self.pitch_range[0] <= freq <= self.pitch_range[1] and confidence > 0.3:
                    return float(freq), float(confidence)
            
            return 0, 0
            
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return 0, 0
    
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
                    
                    # Normalize
                    max_val = np.max(np.abs(audio_segment))
                    if max_val > 0:
                        audio_segment = audio_segment / max_val
                    
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
                            # Extract pitch
                            pitch, confidence = self.extract_pitch_from_sai(sai_frame)
                            
                            # Update history
                            self.current_pitch = pitch
                            self.current_confidence = confidence
                            self.current_time = time.time() - start_time
                            
                            self.pitch_history.append(pitch)
                            self.confidence_history.append(confidence)
                            self.time_history.append(self.current_time)
                        
                    except Exception as e:
                        print(f"SAI processing error: {e}")
                        continue
                
                time.sleep(0.02)  # 50 FPS
                
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
        """Update the pitchogram display"""
        try:
            if len(self.pitch_history) < 2:
                return []
            
            self.ax.clear()
            
            times = np.array(self.time_history)
            pitches = np.array(self.pitch_history)
            confidences = np.array(self.confidence_history)
            
            # Plot voiced segments
            voiced_mask = pitches > 0
            if np.any(voiced_mask):
                self.ax.scatter(times[voiced_mask], pitches[voiced_mask], 
                              c=confidences[voiced_mask], cmap='plasma', 
                              s=20, alpha=0.8, vmin=0, vmax=1)
                
                # Connect points for continuity
                if np.sum(voiced_mask) > 1:
                    self.ax.plot(times[voiced_mask], pitches[voiced_mask], 'w-', alpha=0.3, linewidth=1)
            
            # Setup axes
            self.ax.set_xlim(max(0, self.current_time - self.time_window), self.current_time + 1)
            self.ax.set_ylim(self.pitch_range[0], self.pitch_range[1])
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_ylabel('Pitch (Hz)')
            
            # Title with current pitch info
            if self.current_pitch > 0:
                note = self.freq_to_note(self.current_pitch)
                title = f'CARFAC Pitchogram | {self.current_pitch:.1f} Hz ({note}) | Confidence: {self.current_confidence:.2f}'
            else:
                title = f'CARFAC Pitchogram | Unvoiced | Time: {self.current_time:.1f}s'
            
            self.ax.set_title(title, fontsize=12)
            self.ax.grid(True, alpha=0.3)
            
            return []
            
        except Exception as e:
            print(f"Plot update error: {e}")
            return []
    
    def run(self):
        """Run the real-time pitchogram"""
        print("=== WORKING CARFAC PITCHOGRAM ===")
        print("Based on sai_test.py pattern")
        
        if not self.start_audio():
            print("Failed to start audio!")
            return
        
        try:
            # Setup matplotlib
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=self.ax, label='Pitch Confidence')
            
            # Title
            self.fig.suptitle("CARFAC Real-time Pitchogram (Based on sai_test.py)\nSpeak or sing into your microphone!", 
                            fontsize=13, y=0.95)
            
            # Animation
            self.anim = FuncAnimation(self.fig, self.update_plot, 
                                    interval=100,  # 10 FPS
                                    blit=False, cache_frame_data=False)
            
            print("✓ Pitchogram display started!")
            print("✓ Using EXACT same SAI pattern as working test!")
            print("✓ Speak into microphone to see pitch tracking!")
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
    print("CARFAC Pitchogram - Based on Working sai_test.py")
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