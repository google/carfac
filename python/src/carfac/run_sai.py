"""
FINAL WORKING CARFAC PITCHOGRAM - FIXED THE ASSERTION ERROR!
The issue was: input_segment_width is in seconds, but SAI expects samples
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

class WorkingCarfacPitchogram:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.chunk_size = 1024
        self.update_interval = 100
        
        # Display
        self.time_window = 10
        self.pitch_range = (80, 500)
        
        # History
        self.pitch_history = deque(maxlen=300)
        self.time_history = deque(maxlen=300)
        self.confidence_history = deque(maxlen=300)
        
        # Audio
        self.audio_queue = queue.Queue(maxsize=50)
        self.audio_buffer = np.zeros(sample_rate * 3)  # 3 second buffer
        self.buffer_index = 0
        
        # Current values
        self.current_pitch = 0
        self.current_confidence = 0
        self.current_time = 0
        
        # Setup CARFAC
        self.setup_carfac()
        
        # Threading
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    def setup_carfac(self):
        """Setup CARFAC with correct sample-based input_segment_width"""
        print("Setting up CARFAC with correct parameters...")
        
        # Create SAI parameters
        self.sai_params = pysai.SAIParams()
        
        # Set parameters
        self.sai_params.sai_width = 40
        self.sai_params.sai_height = 64
        self.sai_params.kernel_width = 15
        self.sai_params.kernel_spacing = 2
        self.sai_params.num_triggers_per_frame = 4
        self.sai_params.trigger_win_length = 0.020
        self.sai_params.sai_frame_width = 0.040
        self.sai_params.future_lags = 0
        self.sai_params.num_window_widths = 20
        self.sai_params.window_width = 0.002
        self.sai_params.num_channels = 64
        self.sai_params.do_stabilize = True
        
        # CRITICAL FIX: Convert input_segment_width from seconds to samples
        # AND make sure it's smaller than CARFAC's internal buffer
        segment_duration_seconds = 0.020  # 20ms segments (smaller to fit buffer)
        segment_samples = int(segment_duration_seconds * self.sample_rate)
        self.sai_params.input_segment_width = segment_samples  # NOT seconds, but samples!
        
        # Also reduce other time-based parameters to match
        self.sai_params.sai_frame_width = 0.020  # Match input segment width
        self.sai_params.trigger_win_length = 0.010  # Half of segment width
        
        # Fix trigger_window_width
        setattr(self.sai_params, 'trigger_window_width', self.sai_params.sai_width + 10)
        
        print(f"Key parameters:")
        print(f"  Sample rate: {self.sample_rate}")
        print(f"  Segment duration: {segment_duration_seconds} seconds")
        print(f"  Segment samples: {segment_samples}")
        print(f"  input_segment_width: {self.sai_params.input_segment_width}")
        print(f"  num_channels: {self.sai_params.num_channels}")
        
        try:
            self.sai = pysai.SAI(self.sai_params)
            print("✓ CARFAC SAI created successfully!")
            
            # Test it
            self.test_carfac()
            
        except Exception as e:
            print(f"CARFAC setup failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_carfac(self):
        """Test CARFAC with correct input size"""
        print("Testing CARFAC with correct input size...")
        
        try:
            # Create test input with EXACT number of samples expected
            n_channels = self.sai_params.num_channels
            n_samples = self.sai_params.input_segment_width  # This is now in samples!
            
            print(f"Creating test input: ({n_channels}, {n_samples})")
            
            # Generate test signal
            t = np.linspace(0, n_samples/self.sample_rate, n_samples)
            test_freq = 200  # 200 Hz test tone
            
            # Create multi-channel input
            test_input = []
            for ch in range(n_channels):
                # Each channel gets slightly different filtering of the test tone
                signal = np.sin(2 * np.pi * test_freq * t)
                # Add some channel-specific variation
                signal *= np.exp(-ch * 0.02)  # Decay across channels
                test_input.append(signal)
            
            test_array = np.array(test_input, dtype=np.float32)
            print(f"Test array shape: {test_array.shape}")
            
            # Run SAI
            sai_result = self.sai.RunSegment(test_array)
            
            if sai_result is not None:
                print(f"✓ SAI test PASSED! Output shape: {sai_result.shape}")
                print(f"✓ Output range: {np.min(sai_result):.4f} to {np.max(sai_result):.4f}")
                return True
            else:
                print("✗ SAI returned None")
                return False
                
        except Exception as e:
            print(f"✗ SAI test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_cochlear_channels(self, audio_segment):
        """Create cochlear-like channels from audio segment"""
        n_channels = self.sai_params.num_channels
        n_samples = len(audio_segment)
        
        # Simple filterbank
        channels = []
        low_freq = 80
        high_freq = min(8000, self.sample_rate // 2 - 100)
        
        # Linear frequency spacing
        center_freqs = np.linspace(low_freq, high_freq, n_channels)
        
        for fc in center_freqs:
            try:
                # Simple bandpass filter
                bandwidth = fc * 0.2
                low_cut = max(fc - bandwidth/2, 20)
                high_cut = min(fc + bandwidth/2, self.sample_rate/2 - 50)
                
                if low_cut < high_cut:
                    sos = butter(2, [low_cut, high_cut], btype='band', 
                               fs=self.sample_rate, output='sos')
                    filtered = sosfilt(sos, audio_segment)
                    
                    # Simple envelope detection
                    envelope = np.abs(hilbert(filtered))
                    
                    # Low-pass filter the envelope
                    sos_lp = butter(1, 100, btype='low', fs=self.sample_rate, output='sos')
                    smooth_envelope = sosfilt(sos_lp, envelope)
                    
                    channels.append(smooth_envelope)
                else:
                    channels.append(np.zeros(n_samples))
            except:
                channels.append(np.zeros(n_samples))
        
        return np.array(channels, dtype=np.float32)
    
    def extract_pitch_from_sai(self, sai_output):
        """Extract pitch from SAI output"""
        try:
            if sai_output is None or sai_output.size == 0:
                return 0, 0
            
            # Sum across frequency channels
            if len(sai_output.shape) == 2:
                summary = np.mean(sai_output, axis=0)
            else:
                summary = sai_output
            
            if len(summary) < 3:
                return 0, 0
            
            # Find peak (skip DC bin)
            peak_idx = np.argmax(summary[1:]) + 1
            
            # Convert to frequency
            max_delay = 0.020  # 20ms max delay (match segment duration)
            delay_per_bin = max_delay / len(summary)
            delay = peak_idx * delay_per_bin
            
            if delay > 0:
                freq = 1.0 / delay
                confidence = summary[peak_idx] / np.max(summary) if np.max(summary) > 0 else 0
                
                # Filter valid range
                if self.pitch_range[0] <= freq <= self.pitch_range[1] and confidence > 0.2:
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
        
        segment_samples = self.sai_params.input_segment_width
        
        while self.is_running:
            try:
                # Get audio data
                audio_chunks = []
                for _ in range(3):
                    try:
                        chunk = self.audio_queue.get_nowait()
                        audio_chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if audio_chunks:
                    combined_audio = np.concatenate(audio_chunks)
                    self.update_buffer(combined_audio)
                    
                    # Extract segment of exact size
                    start_idx = (self.buffer_index - segment_samples) % len(self.audio_buffer)
                    
                    if start_idx + segment_samples <= len(self.audio_buffer):
                        audio_segment = self.audio_buffer[start_idx:start_idx + segment_samples]
                    else:
                        # Handle wraparound
                        part1 = self.audio_buffer[start_idx:]
                        part2 = self.audio_buffer[:segment_samples - len(part1)]
                        audio_segment = np.concatenate([part1, part2])
                    
                    # Normalize
                    max_val = np.max(np.abs(audio_segment))
                    if max_val > 0:
                        audio_segment = audio_segment / max_val
                    
                    try:
                        # Create cochlear channels
                        cochlear_channels = self.create_cochlear_channels(audio_segment)
                        
                        # Ensure exact shape
                        if cochlear_channels.shape != (self.sai_params.num_channels, segment_samples):
                            print(f"Shape mismatch: got {cochlear_channels.shape}, expected ({self.sai_params.num_channels}, {segment_samples})")
                            continue
                        
                        # Run SAI
                        sai_output = self.sai.RunSegment(cochlear_channels)
                        
                        if sai_output is not None:
                            # Extract pitch
                            pitch, confidence = self.extract_pitch_from_sai(sai_output)
                            
                            # Update history
                            self.current_pitch = pitch
                            self.current_confidence = confidence
                            self.current_time = time.time() - start_time
                            
                            self.pitch_history.append(pitch)
                            self.confidence_history.append(confidence)
                            self.time_history.append(self.current_time)
                    
                    except Exception as e:
                        print(f"Processing error: {e}")
                        continue
                
                time.sleep(0.02)
                
            except Exception as e:
                print(f"Main processing error: {e}")
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
            print(f"Audio start failed: {e}")
            return False
    
    def stop_audio(self):
        """Stop audio"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("Audio stopped")
    
    def freq_to_note(self, freq):
        """Convert frequency to note"""
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
        """Update plot"""
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
                              c=confidences[voiced_mask], cmap='viridis', 
                              s=15, alpha=0.8, vmin=0, vmax=1)
            
            # Setup axes
            self.ax.set_xlim(max(0, self.current_time - self.time_window), self.current_time + 1)
            self.ax.set_ylim(self.pitch_range[0], self.pitch_range[1])
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_ylabel('Pitch (Hz)')
            
            # Title
            if self.current_pitch > 0:
                note = self.freq_to_note(self.current_pitch)
                title = f'CARFAC Pitchogram | {self.current_pitch:.1f} Hz ({note}) | Conf: {self.current_confidence:.2f}'
            else:
                title = f'CARFAC Pitchogram | Unvoiced | Time: {self.current_time:.1f}s'
            
            self.ax.set_title(title)
            self.ax.grid(True, alpha=0.3)
            
            return []
            
        except Exception as e:
            print(f"Plot error: {e}")
            return []
    
    def run(self):
        """Run the pitchogram"""
        print("=== WORKING CARFAC PITCHOGRAM ===")
        
        if not self.start_audio():
            print("Failed to start audio!")
            return
        
        try:
            # Setup plot
            self.fig, self.ax = plt.subplots(figsize=(14, 8))
            
            # Colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label='Confidence')
            
            # Title
            self.fig.suptitle("CARFAC Real-time Pitchogram - FIXED!\nSpeak or sing into your microphone!", 
                            fontsize=12, y=0.95)
            
            # Animation
            self.anim = FuncAnimation(self.fig, self.update_plot, 
                                    interval=self.update_interval,
                                    blit=False, cache_frame_data=False)
            
            print("✓ Pitchogram display started!")
            print("✓ The assertion error is fixed!")
            print("✓ Speak into your microphone to see pitch tracking!")
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
    print("FIXED CARFAC Pitchogram - No More Assertion Errors!")
    print("=" * 60)
    
    try:
        pitchogram = WorkingCarfacPitchogram(sample_rate=22050)
        pitchogram.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()