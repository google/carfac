"""Tests for carfac."""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from google3.testing.pybase import googletest
import google3.third_party.carfac.python.np.carfac as carfac

# Note some of these tests create plots for easier comparison to the results
# in Dick Lyon's Human and Machine Hearing.  The plots are stored in /tmp, and
# the easiest way to see them is to run the test on your machine:


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


class CarfacUtilityTest(googletest.TestCase):

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
    for l in [cf-bw/2, cf+bw/2]:
      plt.plot([l, l], [a[2], a[3]], 'r--')
    plt.plot([a[0], a[1]], [amplitude-bw_level, amplitude-bw_level], 'b--')
    plt.savefig('/tmp/peak_response.png')


class CarfacTest(googletest.TestCase):

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
    carfac_filters = carfac.design_filters(carfac.CarParams(),
                                           16000,
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
    carfac.carfac_init(cfp)

    n_points = 2**14
    impulse_response = None
    for i in range(n_points):
      # ??? Not sure how this works without state being recursed.
      car_out, _ = carfac.car_step(
          int(i == 0),
          cfp.ears[0].car_coeffs,
          cfp.ears[0].car_state,
          linear=True)
      if impulse_response is None:  # Allocate now when number of channels known
        impulse_response = np.zeros((n_points * 2, len(car_out)))
      impulse_response[i, :] = car_out

    spectrum = np.fft.rfft(impulse_response, axis=0)
    db_spectrum = 20 * np.log10(abs(spectrum))

    db_spectrum[db_spectrum < -20] = -20
    plt.clf()
    plt.imshow(db_spectrum, aspect='auto', vmin=-20, origin='lower')
    plt.xlabel('Channel Number')
    plt.ylabel('Frequency bin')
    plt.colorbar()
    plt.savefig('/tmp/car_freq_resp.png')

    # Test: check overall frequency response of a cascade of CAR filters.
    # Match Figure 16.6 of Lyon's book
    spectrum_freqs = np.arange(n_points + 1) / n_points * cfp.fs / 2.0

    tests = [
        [10, 5604, 39.3, 705.6, 7.9],
        [20, 3245, 55.6, 429.8, 7.6],
        [30, 1809, 60.4, 248.1, 7.3],
        [40, 965, 59.2, 138.7, 7.0],
        [50, 477, 52.8, 74.8, 6.4],
        [60, 195, 39.0, 37.5, 5.2],
        [70, 31, 9.4, 15.1, 2.1],
    ]
    for (channel, correct_cf, correct_gain, correct_bw, correct_q) in tests:
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
    present = (1 - np.sign(np.sin(omega0 * t + np.pi / 2))) / 2
    stim_num = present * np.floor((t / 0.04))
    amplitude = 0.09 * 2**stim_num
    omega = 2 * np.pi * test_freq * present

    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)

    # mn = 0
    phase = 0
    quad_sin = np.sin(omega * t + phase)
    quad_cos = np.cos(omega * t + phase)
    x_in = quad_sin * amplitude
    neuro_output = carfac.ihc_model_run(x_in, fs)

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
        [2.5911196810057446, 714.0938898551138],
        [4.587200409965957, 962.3202339386279],
        [6.764675603953783, 1141.99852325262],
        [8.641429285777633, 1232.3632781905794],
        [9.991694871229491, 1260.4300821170075],
        [10.580712732624507, 1248.8208518862057],
    ]
    blip_maxes, blip_ac = self.run_ihc(300)
    for i in range(len(test_results)):
      max_val, ac = test_results[i]
      self.assertAlmostEqual(blip_maxes[i], max_val, delta=max_val / 1000)
      self.assertAlmostEqual(blip_ac[i], ac, delta=ac / 1000)

    test_results = [
        [1.4056998439907815, 234.1237654636976],
        [2.781740411967144, 316.9254915147316],
        [4.763687397978764, 376.9015111186829],
        [6.967957496695092, 407.98011578429436],
        [8.954049284161007, 420.217293829838],
        [10.426862360234844, 422.57040237627706],
    ]
    blip_maxes, blip_ac = self.run_ihc(3000)
    for i in range(len(test_results)):
      max_val, ac = test_results[i]
      self.assertAlmostEqual(blip_maxes[i], max_val, delta=max_val / 1000)
      self.assertAlmostEqual(blip_ac[i], ac, delta=ac / 1000)

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

    agc_input = np.zeros(cfp.n_ch)
    test_channel = 40
    n_points = 16384
    num_stages = cfp.agc_params.n_stages
    agc_response = np.zeros((num_stages, n_points, cfp.n_ch))
    num_outputs = 0
    for i in range(n_points):
      agc_input[test_channel] = 100
      agc_state, agc_updated = carfac.agc_step(agc_input,
                                               cfp.ears[0].agc_coeffs,
                                               cfp.ears[0].agc_state)
      if agc_updated:
        cfp.ears[0].AGC_state = agc_state
        for stage in range(num_stages):
          agc_response[stage, num_outputs, :] = agc_state[stage].agc_memory
        num_outputs += 1

    # Truncate the response (since we decimated and didn't get an output
    # every step)
    agc_response = agc_response[:, :num_outputs, :]

    # Test: Plot spatial response to match Figure 19.7
    plt.clf()
    plt.plot(agc_response[:, -1, :].T)
    plt.title('Steady state spatial response')
    plt.legend([f'Stage {i}' for i in range(4)])
    plt.savefig('/tmp/agc_steady_state_response.png')

    results = {
        0: [39.033165860723955, 8.359763507183851, 9.598703110909966],
        1: [39.201534368313595, 4.083375909555795, 9.019020871287545],
        2: [39.37440368635895, 1.8782559016120712, 8.219043025392182],
        3: [39.56595707533404, 0.71235061675261, 6.994498097385417],
    }
    for i in range(agc_response.shape[0]):
      # Icky hack, but it works. Find_peak_response wants to find the width at a
      # fixed level (3dB) below the peak.  We call this function twice: the
      # first time with a small width so we can get the peak amplitude.
      # And then a second time with the amplitude that is 50% of the peak.
      [cf, amp, bw] = find_peak_response(
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], .1)
      [cf, amp, bw] = find_peak_response(
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], amp / 2)

      print(f'AGC Stage {i}: Peak at channel {cf}, value is {amp},'
            f'width is {bw} channels')

      expected_cf, expected_amplitude, expected_bw = results[i]
      self.assertAlmostEqual(cf, expected_cf)
      self.assertAlmostEqual(amp, expected_amplitude)
      self.assertAlmostEqual(bw, expected_bw)

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
        cfp, impulse, open_loop=1, linear_car=False)

    initial_freq_response = np.fft.rfft(bm_initial[:1024, :, 0], axis=0)
    initial_freq_response = 20 * np.log10(np.abs(initial_freq_response))

    final_freq_response = np.fft.rfft(bm_final[:1024, :, 0], axis=0)
    final_freq_response = 20 * np.log10(np.abs(final_freq_response))

    # Match Figure 19.9(right) of Lyon's book
    freqs = fs / initial_freq_response.shape[0] * np.arange(
        initial_freq_response.shape[0])
    num_bins = 512

    plt.clf()
    plt.semilogx(freqs[1:num_bins],
                 initial_freq_response[1:num_bins, ::], ':')
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

    results = {
        250: [64, 237.48444306826303, 0.2641456758996128],
        500: [58, 479.1199320357932, 1.0027149596014056],
        1000: [49, 1027.6373544145267, 7.340397337585955],
        2000: [38, 2194.784869601033, 31.65943156233851],
        4000: [28, 4069.8800109530985, 27.253526059781723],
        8000: [16, 8101.940012284346, 13.898859812420024],
        16000: [2, 16547.219145183426, 3.590521826511676],
    }
    for cf in results:
      c = find_closest_channel([f[0] for f in final_resps], cf)
      print(f'Channel {c} has CF of {initial_resps[c][0]} and an '
            f'adaptation change of {initial_resps[c][1]-final_resps[c][1]}dB.')
      expected_c, expected_cf, expected_change = results[cf]
      self.assertEqual(c, expected_c)
      self.assertAlmostEqual(initial_resps[c][0], expected_cf)
      self.assertAlmostEqual(initial_resps[c][1] - final_resps[c][1],
                             expected_change)


if __name__ == '__main__':
  googletest.main()
