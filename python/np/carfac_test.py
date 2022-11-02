"""Tests for carfac."""

import math
from typing import List, Tuple

from absl.testing import absltest
import matplotlib.pyplot as plt
import numpy as np

from . import carfac

# Note some of these tests create plots for easier comparison to the results
# in Dick Lyon's Human and Machine Hearing.  The plots are stored in /tmp, and
# the easiest way to see them is to run the test on your machine; or in a
# Colab such as google3/third_party/carfac/python/np/CARFAC_Testing.ipynb


def linear_interp(x: np.ndarray, pos: float) -> float:
  if pos <= 0:
    return x[0]
  elif pos >= x.shape[0] - 1:
    return x[-1]
  else:
    i = int(pos)
    frac = pos - i
    return x[i] * (1 - frac) + x[i + 1] * frac


# From:
# https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
def quadratic_peak_interpolation(alpha: float, beta: float,
                                 gamma: float) -> Tuple[float, float]:
  location = 1 / 2 * (alpha - gamma) / (alpha - 2 * beta + gamma)
  amplitude = beta - 1.0 / 4 * (alpha - gamma) * location
  return location, amplitude


def find_peak_response(freqs, db_gains, bw_level=3):
  """Returns center frequency, amplitude at this point, and the 3dB width."""
  peak_bin = np.argmax(db_gains)
  peak_frac, amplitude = quadratic_peak_interpolation(db_gains[peak_bin - 1],
                                                      db_gains[peak_bin],
                                                      db_gains[peak_bin + 1])
  cf = linear_interp(freqs, peak_bin + peak_frac)

  freqs_3db = find_zero_crossings(freqs, db_gains - amplitude + bw_level)
  if len(freqs_3db) >= 2:
    return cf, amplitude, freqs_3db[1] - freqs_3db[0]
  return cf, amplitude, 0


def find_zero_crossings(x: np.ndarray, y: np.ndarray) -> List[int]:
  locs = list(np.where(y[1:] * y[:-1] < 0)[0])

  def interpolate_zc(x, y, i):
    a = y[i]
    b = y[i + 1]
    if a > b:
      a = -a
      b = -b
    frac = -a / (b - a)
    return x[i] * (1 - frac) + x[i + 1] * frac

  return [interpolate_zc(x, y, i) for i in locs]


class CarfacUtilityTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Simple frequency response for testing.
    def test_peak(f):
      characteristic_frequency = 1000
      bandwidth = 8
      bw2 = bandwidth / 2.0
      low_freq = characteristic_frequency - characteristic_frequency / bw2
      high_freq = characteristic_frequency + characteristic_frequency / bw2

      scale = ((characteristic_frequency - low_freq) *
               (characteristic_frequency - high_freq))
      return (f - low_freq) * (f - high_freq) / scale

    # Don't use interval increment so we need to interpolate
    self.test_freqs = np.arange(800, 1200, .37)
    self.test_resp = test_peak(self.test_freqs)

  def test_linear_interp(self):
    # Test: One point
    num = 10
    test_data = np.arange(num)**2
    # First Should be exactly 4, since it's a real point, and second is close
    # to 5 since it is between samples.
    self.assertAlmostEqual(linear_interp(test_data, math.sqrt(4)), 4)
    self.assertAlmostEqual(linear_interp(test_data, math.sqrt(5)), 5.1803398874)

  def test_quadractic_peak_interpolation(self):
    max_bin = np.argmax(self.test_resp)
    print(f'max_bin at {max_bin}, with value {self.test_freqs[max_bin]}')
    self.assertEqual(max_bin, 541)
    location, amplitude = quadratic_peak_interpolation(
        self.test_resp[max_bin - 1], self.test_resp[max_bin],
        self.test_resp[max_bin + 1])
    print(f'Exact peak at bin {location}, with value {amplitude}')
    self.assertAlmostEqual(amplitude, 1.0)
    best_f = linear_interp(self.test_freqs, max_bin + location)
    print(f'Peak value is at {best_f} Hz.')
    self.assertAlmostEqual(best_f, 1000)

    # For viewing test example.
    plt.clf()
    plt.plot(self.test_freqs, self.test_resp)
    plt.plot(best_f, amplitude, 'x')
    plt.savefig('/tmp/quadractic_interpolation.png')

  def test_peak_response(self):
    bw_level = 0.5
    cf, amplitude, bw = find_peak_response(
        self.test_freqs, self.test_resp, bw_level=bw_level)
    print(f'Filter cf is {cf}, with peak amplitude of {amplitude} and bw {bw}.')
    self.assertAlmostEqual(cf, 1000)
    self.assertAlmostEqual(amplitude, 1)
    self.assertAlmostEqual(bw, 353.5532375878539)

    # For verifying peak characterization tests.
    plt.clf()
    plt.plot(self.test_freqs, self.test_resp)
    plt.plot(cf, amplitude, 'x')
    a = plt.axis()
    for l in [cf - bw / 2, cf + bw / 2]:
      plt.plot([l, l], [a[2], a[3]], 'r--')
    plt.plot([a[0], a[1]], [amplitude - bw_level, amplitude - bw_level], 'b--')
    plt.savefig('/tmp/peak_response.png')


class CarfacTest(absltest.TestCase):

  def test_hz_to_erb(self):
    # Test: Simple, should asymptote to 9.2645
    self.assertAlmostEqual(100 / carfac.hz_to_erb(100), 2.8173855225827538)
    self.assertAlmostEqual(1000 / carfac.hz_to_erb(1000), 7.53926070009575)
    self.assertAlmostEqual(2000 / carfac.hz_to_erb(2000), 8.313312106676422)
    self.assertAlmostEqual(4000 / carfac.hz_to_erb(4000), 8.763166657903502)
    self.assertAlmostEqual(8000 / carfac.hz_to_erb(8000), 9.006858722917503)
    self.assertAlmostEqual(16000 / carfac.hz_to_erb(16000), 9.133858986918032)

  def test_carfac_design(self):
    # Test: Simple.  But where do the divide by zeros come from????
    carfac_filters = carfac.design_filters(carfac.CarParams(), 16000,
                                           math.pi * np.arange(1, 5) / 5.)
    print(carfac_filters)

  def test_carfac_detect(self):
    # Test: Simple
    carfac.ihc_detect(10)
    carfac.ihc_detect(np.array((1e6, 10.0, .0)))

  def test_design_fir_coeffs(self):
    carfac.design_fir_coeffs(3, 1, 1, 1)
    carfac.design_fir_coeffs(5, 1, 1, 1)

  def test_car_freq_response(self):
    cfp = carfac.design_carfac()
    carfac.carfac_init(cfp)

    # Show impulse response for just the CAR Filter bank.
    n_points = 2**14
    fft_len = n_points * 2  # Lots of zero padding for good freq. resolution.
    impulse_response = np.zeros((fft_len, cfp.n_ch))
    for i in range(n_points):
      [car_out, cfp.ears[0].car_state] = carfac.car_step(
          int(i == 0),
          cfp.ears[0].car_coeffs,
          cfp.ears[0].car_state,
          linear=True)
      impulse_response[i, :] = car_out

    spectrum = np.fft.rfft(impulse_response, axis=0)
    db_spectrum = 20 * np.log10(abs(spectrum) + 1e-50)

    db_spectrum[db_spectrum < -20] = -20
    plt.clf()
    plt.imshow(db_spectrum, aspect='auto', vmin=-20, origin='lower')
    plt.xlabel('Channel Number')
    plt.ylabel('Frequency bin')
    plt.colorbar()
    plt.savefig('/tmp/car_freq_resp.png')

    # Test: check overall frequency response of a cascade of CAR filters.
    # Match Figure 16.6 of Lyon's book
    spectrum_freqs = np.arange(n_points + 1) / fft_len * cfp.fs

    expected = [
        [10, 5604, 39.3, 705.6, 7.9],
        [20, 3245, 55.6, 429.8, 7.6],
        [30, 1809, 60.4, 248.1, 7.3],
        [40, 965, 59.2, 138.7, 7.0],
        [50, 477, 52.8, 74.8, 6.4],
        [60, 195, 39.0, 37.5, 5.2],
        [70, 31, 9.4, 15.1, 2.1],
    ]
    for (channel, correct_cf, correct_gain, correct_bw, correct_q) in expected:
      cf, amplitude, bw = find_peak_response(spectrum_freqs,
                                             db_spectrum[:, channel])
      gain = round(10 * amplitude) / 10
      q = round(10 * cf / bw) / 10
      cf = round(cf)
      bw = round(10 * bw) / 10
      print(f'{channel}: cf is {cf} Hz, peak gain is '
            f'{gain} dB, 3 dB bandwidth is {bw} Hz '
            f'(Q = {q})')
      self.assertAlmostEqual(cf, correct_cf)
      self.assertAlmostEqual(gain, correct_gain)
      self.assertAlmostEqual(bw, correct_bw)
      self.assertAlmostEqual(q, correct_q)

  def run_ihc(self, test_freq=300):
    fs = 40000
    sampling_interval = 1 / fs
    tmax = 0.28  # a half second

    t = np.arange(0, tmax, sampling_interval)

    # test signal, ramping square wave gating a sinusoid:
    omega0 = 2 * np.pi * 25  # for making a square wave envelope
    # start at 0.09, 6 db below mean response threshold
    present = (1 - np.sign(np.sin(omega0 * t + np.pi / 2 + 1e-6))) / 2
    stim_num = present * np.floor((t / 0.04))
    amplitude = 0.09 * 2**stim_num
    omega = 2 * np.pi * test_freq

    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)

    quad_sin = np.sin(omega * t) * present
    quad_cos = np.cos(omega * t) * present
    x_in = quad_sin * amplitude
    neuro_output = np.zeros(x_in.shape)
    for i in range(x_in.shape[0]):
      [ihc_out,
       cfp.ears[0].ihc_state] = carfac.ihc_step(x_in[i], cfp.ears[0].ihc_coeffs,
                                                cfp.ears[0].ihc_state)
      neuro_output[i] = ihc_out[0]

    plt.clf()
    plt.plot(t, neuro_output)
    plt.xlabel('Seconds')
    plt.title(f'IHC Response for tone blips at {test_freq}Hz')
    plt.savefig(f'/tmp/ihc_response_{test_freq}Hz.png')

    blip_maxes = []
    blip_ac = []
    for i in range(1, 7):
      blip = neuro_output * (stim_num == i)
      blip_max = blip[np.argmax(blip)]
      carrier_power = np.sum(blip * quad_sin)**2 + np.sum(blip * quad_cos)**2
      print(f'Blip {i}: Max of {blip_max}, '
            f'AC level is {np.sqrt(carrier_power)}')
      blip_maxes.append(blip_max)
      blip_ac.append(np.sqrt(carrier_power))
    return blip_maxes, blip_ac

  def test_ihc(self):
    test_results = [
        [2.591120, 714.093923],
        [4.587200, 962.263411],
        [6.764676, 1141.981740],
        [8.641429, 1232.411638],
        [9.991695, 1260.430082],
        [10.580713, 1248.820852],
    ]
    blip_maxes, blip_ac = self.run_ihc(300)
    for i in range(len(test_results)):
      max_val, ac = test_results[i]
      self.assertAlmostEqual(blip_maxes[i], max_val, delta=max_val / 10000)
      self.assertAlmostEqual(blip_ac[i], ac, delta=ac / 10000)

    test_results = [  # From the Matlab test
        [1.405700, 234.125387],
        [2.781740, 316.707076],
        [4.763687, 376.773748],
        [6.967957, 407.915892],
        [8.954049, 420.217294],
        [10.426862, 422.570402],
    ]
    blip_maxes, blip_ac = self.run_ihc(3000)
    for i in range(len(test_results)):
      max_val, ac = test_results[i]
      self.assertAlmostEqual(blip_maxes[i], max_val, delta=max_val / 10000)
      self.assertAlmostEqual(blip_ac[i], ac, delta=ac / 10000)

  def test_shift_right(self):
    # Test: By direct comparison to the 5 cases in the
    # Matab Spatial_Smooth
    # function.  Test these 5 cases, for these exact values.
    expected = {
        -2: [2, 3, 4, 5, 6, 6, 5],
        -1: [1, 2, 3, 4, 5, 6, 6],
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [0, 0, 1, 2, 3, 4, 5],
        2: [0, 1, 0, 1, 2, 3, 4]
    }
    for amount in expected:
      result = carfac.shift_right(np.arange(7), amount)
      print(f'{amount}: {result}')
      self.assertEqual(list(result), expected[amount])

  def test_agc_steady_state(self):
    # Test: Steady state response
    # Analagous to figure 19.7
    cfp = carfac.design_carfac()
    cf = carfac.carfac_init(cfp)

    test_channel = 40
    agc_input = np.zeros(cfp.n_ch)
    agc_input[test_channel] = 100  # Arbitray spatial impulse size.
    num_stages = cfp.agc_params.n_stages
    decim = cfp.agc_params.decimation[0]
    n_points = 2**14  # 2**13 doesn't converge to 7-digit accuracy.
    agc_response = np.zeros((num_stages, n_points // decim, cfp.n_ch))
    num_outputs = 0
    for i in range(n_points):
      [agc_updated,
       agc_state] = carfac.agc_step(agc_input, cfp.ears[0].agc_coeffs,
                                    cfp.ears[0].agc_state)
      cfp.ears[0].agc_state = agc_state  # Not really necessary in np.
      if agc_updated:
        for stage in range(num_stages):
          agc_response[stage, num_outputs, :] = agc_state[stage].agc_memory
        num_outputs += 1

    self.assertEqual(num_outputs, n_points // decim)

    # Test: Plot spatial response to match Figure 19.7
    plt.clf()
    plt.plot(agc_response[:, -1, :].T)
    plt.title('Steady state spatial response')
    plt.legend([f'Stage {i}' for i in range(4)])
    plt.savefig('/tmp/agc_steady_state_response.png')

    expected_results = {  # From Matlab test:
        0: [39.033166, 8.359763, 9.598703],
        1: [39.201534, 4.083376, 9.019020],
        2: [39.374404, 1.878256, 8.219043],
        3: [39.565957, 0.712351, 6.994498],
    }
    for i in range(num_stages):
      # Find_peak_response wants to find the width at a fixed level (3 dB)
      # below the peak.  We call this function twice: the first time with
      # a simple estimate of the max; then a second time with a threshold
      # that is 50% down from the interpolated peak amplitude to get width.
      amp = np.max(agc_response[i, -1, :])
      [cf, amp, bw] = find_peak_response(
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], amp / 2)
      [cf, amp, bw] = find_peak_response(
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], amp / 2)

      print(f'AGC Stage {i}: Peak at channel {cf}, value is {amp},'
            f'width is {bw} channels')

      expected_cf, expected_amplitude, expected_bw = expected_results[i]
      self.assertAlmostEqual(cf, expected_cf, places=5)
      self.assertAlmostEqual(amp, expected_amplitude, places=5)
      self.assertAlmostEqual(bw, expected_bw, places=5)

  def test_whole_carfac(self):
    # Test: Make sure that the AGC adapts to a tone. Test with open-loop impulse
    # response.

    fs = 22050.0
    fp = 1000.0  # Probe tone
    t = np.arange(0, 2, 1 / fs)  # 2s of tone
    sinusoid = 1e-1 * np.sin(2 * np.pi * t * fp)

    t = np.arange(0, 0.5, 1 / fs)
    impulse = np.zeros(t.shape)
    impulse[0] = 1e-4

    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)

    _, cfp, bm_initial, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=1, linear_car=True)

    carfac.run_segment(cfp, sinusoid, open_loop=0)

    # Let filter ringing die, linear_car=true overflowed car
    carfac.run_segment(cfp, 0 * impulse, open_loop=1, linear_car=False)

    _, cfp, bm_final, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=1, linear_car=True)

    fft_len = 2048  # Because 1024 is too sensitive for some reason.
    num_bins = fft_len // 2 + 1  # Per np.fft.rfft.
    freqs = fs / fft_len * np.arange(num_bins)

    initial_freq_response = np.fft.rfft(bm_initial[:fft_len, :, 0], axis=0)
    initial_freq_response = 20 * np.log10(np.abs(initial_freq_response) + 1e-50)

    final_freq_response = np.fft.rfft(bm_final[:fft_len, :, 0], axis=0)
    final_freq_response = 20 * np.log10(np.abs(final_freq_response) + 1e-50)

    # Match Figure 19.9(right) of Lyon's book
    plt.clf()
    plt.semilogx(freqs[1:num_bins], initial_freq_response[1:num_bins, ::], ':')
    # https://stackoverflow.com/questions/24193174/reset-color-cycle-in-matplotlib
    plt.gca().set_prop_cycle(None)
    plt.semilogx(freqs[1:num_bins], final_freq_response[1:num_bins, ::])
    plt.ylabel('dB')
    plt.xlabel('FFT Bin')
    plt.title('Initial (dotted) vs. Adapted at 1kHz (solid) '
              'Frequency Response')
    plt.ylim(-100, -15)
    plt.savefig('/tmp/whole_carfac_response.png')

    initial_resps = [
        find_peak_response(freqs, initial_freq_response[:, i])
        for i in range(71)
    ]
    final_resps = [
        find_peak_response(freqs, final_freq_response[:, i]) for i in range(71)
    ]

    # Test for change in peak gain after adaptation.
    def find_closest_channel(cfs: List[float], desired: float) -> np.ndarray:
      return np.argmin((np.asarray(cfs) - desired)**2)

    results = {  # The Matlab test prints this data block:
        125: [64, 118.944255, 0.186261],
        250: [58, 239.771898, 0.910003],
        500: [49, 514.606412, 7.243568],
        1000: [38, 1099.433179, 31.608529],
        2000: [28, 2038.873929, 27.242882],
        4000: [16, 4058.881505, 13.865787],
        8000: [2, 8289.882476, 3.574972],
    }
    for cf in results:
      c = find_closest_channel([f[0] for f in final_resps], cf)
      expected_c, expected_cf, expected_change = results[cf]
      cf = initial_resps[expected_c][0]
      diff_db = initial_resps[expected_c][1] - final_resps[expected_c][1]
      print(f'Channel {expected_c} has CF of {cf} and an '
            f'adaptation change of {diff_db} dB.')
      self.assertEqual(c, expected_c)
      self.assertAlmostEqual(cf, expected_cf, delta=expected_cf / 10000.0)
      self.assertAlmostEqual(
          diff_db, expected_change, delta=0.01)  # dB change diff.

  def test_delay_buffer(self):
    # Test: Verify simple delay of linear impulse response.

    fs = 22050.0
    t = np.arange(0, 0.1, 1 / fs)  # Short impulse input.
    impulse = np.zeros(t.shape)
    impulse[0] = 1e-4

    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)
    # Run the linear case with small impulse.
    _, cfp, bm_initial, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=True, linear_car=True)
    cfp = carfac.carfac_init(cfp)  # Clear state to zero between runs.
    cfp.ears[0].car_coeffs.use_delay_buffer = True
    _, cfp, bm_delayed, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=True, linear_car=True)
    max_max_rel_error = 0
    for ch in np.arange(cfp.n_ch):
      impresp = bm_initial[:-ch - 1, ch]
      delayed = bm_delayed[ch:-1, ch]
      max_abs = np.amax(np.abs(impresp))
      max_abs_error = np.amax(np.abs(impresp - delayed))
      max_rel = max_abs_error / max_abs
      max_max_rel_error = np.maximum(max_max_rel_error, max_rel)
      self.assertLess(max_rel, 2e-4)  # Needs more tolerance than Matlab. Why?
    print(max_max_rel_error)

    # Run the nonlinear case with a small impulse so not too nonlinear.
    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)
    cfp.ears[0].car_coeffs.use_delay_buffer = False
    _, cfp, bm_initial, _, _ = carfac.run_segment(cfp, impulse)
    cfp = carfac.carfac_init(cfp)  # Clear state to zero between runs.
    cfp.ears[0].car_coeffs.use_delay_buffer = True
    _, cfp, bm_delayed, _, _ = carfac.run_segment(cfp, impulse)
    max_max_rel_error = 0
    for ch in np.arange(cfp.n_ch):
      impresp = bm_initial[:-ch - 1, ch]
      delayed = bm_delayed[ch:-1, ch]
      max_abs = np.amax(np.abs(impresp))
      max_abs_error = np.amax(np.abs(impresp - delayed))
      max_rel = max_abs_error / max_abs
      max_max_rel_error = np.maximum(max_max_rel_error, max_rel)
      self.assertLess(max_rel, 0.025)
    print(max_max_rel_error)


if __name__ == '__main__':
  absltest.main()
