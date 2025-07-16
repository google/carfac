"""Tests for carfac."""

import math
from typing import List, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import matplotlib.pyplot as plt
import numpy as np

from carfac.np import carfac

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
def quadratic_peak_interpolation(
    alpha: float, beta: float, gamma: float
) -> Tuple[float, float]:
  location = 1 / 2 * (alpha - gamma) / (alpha - 2 * beta + gamma)
  amplitude = beta - 1.0 / 4 * (alpha - gamma) * location
  return location, amplitude


def find_peak_response(freqs, db_gains, bw_level=3):
  """Returns center frequency, amplitude at this point, and the 3dB width."""
  peak_bin = np.argmax(db_gains)
  peak_frac, amplitude = quadratic_peak_interpolation(
      db_gains[peak_bin - 1], db_gains[peak_bin], db_gains[peak_bin + 1]
  )
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

      scale = (characteristic_frequency - low_freq) * (
          characteristic_frequency - high_freq
      )
      return (f - low_freq) * (f - high_freq) / scale

    # Don't use interval increment so we need to interpolate
    self.test_freqs = np.arange(800, 1200, 0.37)
    self.test_resp = test_peak(self.test_freqs)

  def test_linear_interp(self):
    # Test: One point
    num = 10
    test_data = np.arange(num) ** 2
    # First Should be exactly 4, since it's a real point, and second is close
    # to 5 since it is between samples.
    self.assertAlmostEqual(linear_interp(test_data, math.sqrt(4)), 4)
    self.assertAlmostEqual(linear_interp(test_data, math.sqrt(5)), 5.1803398874)

  def test_quadractic_peak_interpolation(self):
    max_bin = np.argmax(self.test_resp)
    print(f'max_bin at {max_bin}, with value {self.test_freqs[max_bin]}')
    self.assertEqual(max_bin, 541)
    location, amplitude = quadratic_peak_interpolation(
        self.test_resp[max_bin - 1],
        self.test_resp[max_bin],
        self.test_resp[max_bin + 1],
    )
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
        self.test_freqs, self.test_resp, bw_level=bw_level
    )
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


class CarfacTest(parameterized.TestCase):

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
    carfac_filters = carfac.design_filters(
        carfac.CarParams(), 16000, math.pi * np.arange(1, 5) / 5.0
    )
    print(carfac_filters)

  def test_carfac_detect(self):
    # Test: Simple
    carfac.ihc_detect(10)
    carfac.ihc_detect(np.array((1e6, 10.0, 0.0)))

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
          linear=True,
      )
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

    # Channel, CF, Peak gain, bandwidth, Q
    # Numbers taken directly from the Matlab test.
    expected = [
        [10, 5604, 39.34, 705.6, 7.9],
        [20, 3245, 55.61, 429.8, 7.6],
        [30, 1809, 60.46, 248.1, 7.3],
        [40, 965, 59.18, 138.7, 7.0],
        [50, 477, 52.81, 74.8, 6.4],
        [60, 195, 38.98, 37.5, 5.2],
        [70, 32, 7.95, 14.6, 2.2],
    ]
    for channel, correct_cf, correct_gain, correct_bw, correct_q in expected:
      cf, amplitude, bw = find_peak_response(
          spectrum_freqs, db_spectrum[:, channel]
      )
      # Round and test gain to 2 places since float32 math is a
      # bit different to float64.
      gain = round(amplitude, 2)
      q = round(cf / bw, 1)
      cf = round(cf)
      print(
          f'{channel}: cf is {cf} Hz, peak gain is '
          f'{gain} dB, 3 dB bandwidth is {round(bw, 1)} Hz '
          f'(Q = {q})'
      )
      self.assertAlmostEqual(cf, correct_cf)
      self.assertAlmostEqual(amplitude, correct_gain, delta=0.01)
      self.assertAlmostEqual(bw, correct_bw, delta=0.1)
      self.assertAlmostEqual(q, correct_q)

  def run_ihc(self, test_freq=300, ihc_style='one_cap'):
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

    cfp = carfac.design_carfac(fs=fs, ihc_style=ihc_style)
    cfp = carfac.carfac_init(cfp)
    syn_state = None
    if cfp.syn_params and cfp.syn_params.do_syn:
      syn_state = cfp.ears[0].syn_state

    quad_sin = np.sin(omega * t) * present
    quad_cos = np.cos(omega * t) * present
    x_in = quad_sin * amplitude
    neuro_output = np.zeros(x_in.shape)
    for i in range(x_in.shape[0]):
      [ihc_out, cfp.ears[0].ihc_state, v_recep] = carfac.ihc_step(
          x_in[i], cfp.ears[0].ihc_coeffs, cfp.ears[0].ihc_state
      )
      # Ignore ihc_out and use receptor potential.
      if cfp.syn_params and cfp.syn_params.do_syn:
        [ihc_syn_out, _, syn_state] = carfac.syn_step(
            v_recep, cfp.ears[0].syn_coeffs, syn_state
        )
        neuro_output[i] = ihc_syn_out[0]
      else:
        neuro_output[i] = ihc_out[0]

    plt.figure()
    plt.clf()
    plt.plot(t, neuro_output)
    plt.xlabel('Seconds')
    plt.title(f'IHC Response for tone blips at {test_freq}Hz')
    plt.savefig(f'/tmp/ihc_response_cap_{ihc_style}_{test_freq}Hz.png')
    blip_maxes = []
    blip_ac = []
    for i in range(1, 7):
      blip = neuro_output * (stim_num == i)
      blip_max = blip[np.argmax(blip)]
      carrier_power = (
          np.sum(blip * quad_sin) ** 2 + np.sum(blip * quad_cos) ** 2
      )
      print(
          f'Blip {i}: Max of {blip_max}, AC level is {np.sqrt(carrier_power)}'
      )
      blip_maxes.append(blip_max)
      blip_ac.append(np.sqrt(carrier_power))
    return blip_maxes, blip_ac

  # all test results from the matlab test.
  @parameterized.named_parameters(
      (
          'two_cap_with_syn_300',
          'two_cap_with_syn',
          300,
          [
              [1.055837, 184.180863],
              [3.409906, 483.204136],
              [6.167359, 837.629296],
              [7.096430, 956.279101],
              [7.103324, 927.060415],
              [7.123434, 895.574871],
          ],
      ),
      (
          'two_cap_with_syn_3000',
          'two_cap_with_syn',
          3000,
          [
              [0.167683, 24.929044],
              [0.620939, 74.045598],
              [1.894064, 175.367201],
              [3.541070, 269.322147],
              [4.899921, 303.684269],
              [5.572545, 278.428744],
          ],
      ),
      (
          'two_cap_300',
          'two_cap',
          300,
          [
              [2.026682, 544.901381],
              [3.533259, 756.736631],
              [5.108579, 923.142282],
              [6.423783, 1017.472318],
              [7.454677, 1059.407644],
              [8.231247, 1071.902335],
          ],
      ),
      (
          'two_cap_3000',
          'two_cap',
          3000,
          [
              [0.698303, 93.388172],
              [1.520033, 131.832247],
              [2.660770, 163.287206],
              [3.872406, 182.022912],
              [4.909175, 191.225206],
              [5.666469, 194.912279],
          ],
      ),
      (
          'one_cap_300',
          'one_cap',
          300,
          [
              [2.752913, 721.001685],
              [4.815015, 969.505412],
              [7.062418, 1147.285676],
              [9.138118, 1239.521055],
              [10.969522, 1277.061337],
              [12.516468, 1285.880084],
          ],
      ),
      (
          'one_cap_3000',
          'one_cap',
          3000,
          [
              [1.417657, 234.098558],
              [2.804747, 316.717957],
              [4.802444, 376.787575],
              [7.030791, 408.011707],
              [9.063014, 420.602740],
              [10.634581, 423.674628],
          ],
      ),
  )
  def test_ihc_param(self, cap, freq, test_results):
    blip_maxes, blip_ac = self.run_ihc(freq, ihc_style=cap)
    for i, (max_val, ac) in enumerate(test_results):
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
        2: [0, 1, 0, 1, 2, 3, 4],
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
      [agc_updated, agc_state] = carfac.agc_step(
          agc_input, cfp.ears[0].agc_coeffs, cfp.ears[0].agc_state
      )
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
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], amp / 2
      )
      [cf, amp, bw] = find_peak_response(
          np.arange(agc_response.shape[-1]), agc_response[i, -1, :], amp / 2
      )

      print(
          f'AGC Stage {i}: Peak at channel {cf}, value is {amp},width is'
          f' {bw} channels'
      )

      expected_cf, expected_amplitude, expected_bw = expected_results[i]
      self.assertAlmostEqual(cf, expected_cf, places=5)
      self.assertAlmostEqual(amp, expected_amplitude, places=5)
      self.assertAlmostEqual(bw, expected_bw, places=5)

  def test_stage_g_calculation(self):
    fs = 22050.0
    cfp = carfac.design_carfac(fs=fs)
    # Set to true to save a large number of figures.
    do_plots = False
    # arange goes to just above 1 to ensure 1.0 is tested.
    for undamping in np.arange(0, 1.01, 0.1):
      ideal_g = carfac.design_stage_g(cfp.ears[0].car_coeffs, undamping)
      stage_g = carfac.stage_g(cfp.ears[0].car_coeffs, undamping)
      abs_delta = np.abs(ideal_g - stage_g)
      if do_plots:
        plt.figure()
        plt.plot(ideal_g, 'g-')
        plt.plot(stage_g, 'r.')
        plt.plot(1000 * abs_delta, 'm+')
        plt.title('test stage g calculation')
        plt.xlabel('channel')
        plt.ylabel('Stage gains and 1000*abs(error)')
        plt.savefig(f'/tmp/stage_g_calculation_undamping_{undamping}.png')
      normalized_delta = abs_delta / ideal_g
      for ch in np.arange(cfp.n_ch):
        self.assertLess(
            normalized_delta[ch],
            1e-3,
            f'Failed at channel {ch} for undamping {undamping}.',
        )

  @parameterized.named_parameters(
      ('two_cap', 'two_cap'), ('one_cap', 'one_cap')
  )
  def test_whole_carfac(self, ihc_style):
    # Test: Make sure that the AGC adapts to a tone. Test with open-loop impulse
    # response.

    fs = 22050.0
    fp = 1000.0  # Probe tone
    t = np.arange(0, 2, 1 / fs)  # 2s of tone
    sinusoid = 1e-1 * np.sin(2 * np.pi * t * fp)

    t = np.arange(0, 0.5, 1 / fs)
    impulse = np.zeros(t.shape)
    impulse[0] = 1e-4

    cfp = carfac.design_carfac(fs=fs, ihc_style=ihc_style)
    cfp = carfac.carfac_init(cfp)

    _, cfp, bm_initial, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=1, linear_car=True
    )

    carfac.run_segment(cfp, sinusoid, open_loop=0)

    # Let filter ringing die, linear_car=true overflowed car
    carfac.run_segment(cfp, 0 * impulse, open_loop=1, linear_car=False)

    _, cfp, bm_final, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=1, linear_car=True
    )

    fft_len = 2048  # Because 1024 is too sensitive for some reason.
    num_bins = fft_len // 2 + 1  # Per np.fft.rfft.
    freqs = fs / fft_len * np.arange(num_bins)

    plt.figure()
    plot_channel = 64
    plt.plot(bm_initial[:1500, plot_channel, 0])
    plt.title(f'NP: Channel {plot_channel} output bm_initial')
    plt.savefig('/tmp/whole_carfac_bm_initial.png')
    plt.savefig(f'/tmp/whole_channel_{plot_channel}_response.png')
    print(
        f'Max value of bm_inital channel {plot_channel} is'
        f' {np.max(bm_initial[:1500, plot_channel, 0])} '
    )

    plt.figure()
    plt.semilogy(np.max(bm_initial[:, :, 0], axis=0))
    plt.semilogy(np.max(bm_final[:, :, 0], axis=0))
    plt.ylim(1e-6, 1e-2)
    plt.title('NP: Peak impulse response value')
    plt.xlabel('Channel Number')
    plt.legend(('Initial', 'Final'))
    plt.savefig('/tmp/whole_carfac_peak_response.png')

    print('Maximum output from each channel - Numpy')
    print('Channel: Before / After adaptation')
    for c in range(0, bm_initial.shape[1], 5):
      print(
          f'{c}: ({np.max(bm_initial[:, c, 0])}, {np.max(bm_final[:, c, 0])}),'
      )

    # The following data comes from the Numpy implementation
    max_expected_responses = {  # By channel, pre and post adaptation
        0: (9.487948409514502e-05, 9.490316690485419e-05),
        5: (0.0001728739298414439, 0.00010075179567321992),
        10: (0.0008515723166055977, 0.00034923167361843694),
        15: (0.002147773513570428, 0.0005341274528950696),
        20: (0.0034863983746618032, 0.0005823346124505075),
        25: (0.003892917651683092, 0.000378483231432649),
        30: (0.003512393683195114, 0.00017962464917444995),
        35: (0.002596578560769558, 9.984285850628838e-05),
        40: (0.0016622854163870215, 7.867478345050496e-05),
        45: (0.0009421658469364047, 0.00014724010765066194),
        50: (0.0004423711507115513, 0.00021987651242219705),
        55: (0.00015971883840393275, 0.0001273608619059981),
        60: (4.5056131057208404e-05, 4.183175699987604e-05),
        65: (7.765176633256488e-06, 7.573388744425412e-06),
        70: (5.994126581754244e-07, 5.919053135128626e-07),
    }
    if ihc_style == 'two_cap':
      # The following data comes from the Numpy implementation
      max_expected_responses = {  # By channel, pre and post adaptation
          0: (9.487948409514502e-05, 9.489925609401865e-05),
          5: (0.0001728739298414439, 0.00010618242255033245),
          10: (0.0008515723166055977, 0.0003760562990789565),
          15: (0.002147773513570428, 0.0006027355378328852),
          20: (0.0034863983746618032, 0.000647666208076673),
          25: (0.003892917651683092, 0.0004237077512010596),
          30: (0.003512393683195114, 0.0001927438401829605),
          35: (0.002596578560769558, 0.00010505300902051301),
          40: (0.0016622854163870215, 8.036343425013805e-05),
          45: (0.0009421658469364047, 0.00014942006428250514),
          50: (0.0004423711507115513, 0.00022170255622268033),
          55: (0.00015971883840393275, 0.00012781287725785215),
          60: (4.5056131057208404e-05, 4.1886608749658414e-05),
          65: (7.765176633256488e-06, 7.577093706129464e-06),
          70: (5.994126581754244e-07, 5.920810347298803e-07),
      }

    def err(a, b):
      return (a - b) / b

    for c in max_expected_responses:
      e = err(max_expected_responses[c][0], np.max(bm_initial[:, c, 0]))
      print(f'Pre {c}: {e*100}% error.')
      e = err(max_expected_responses[c][1], np.max(bm_final[:, c, 0]))
      print(f'Post {c}: {e*100}% error.')

    for c in max_expected_responses:  # First check pre adaptation results
      self.assertAlmostEqual(
          max_expected_responses[c][0], np.max(bm_initial[:, c, 0])
      )
    for c in max_expected_responses:  # Then check post adaptation results
      self.assertAlmostEqual(
          max_expected_responses[c][1], np.max(bm_final[:, c, 0])
      )

    initial_freq_response = np.fft.rfft(bm_initial[:fft_len, :, 0], axis=0)
    initial_freq_response = 20 * np.log10(np.abs(initial_freq_response) + 1e-50)

    final_freq_response = np.fft.rfft(bm_final[:fft_len, :, 0], axis=0)
    final_freq_response = 20 * np.log10(np.abs(final_freq_response) + 1e-50)

    # Match Figure 19.9(right) of Lyon's book
    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.semilogx(freqs[1:num_bins], initial_freq_response[1:num_bins, ::1], ':')
    # https://stackoverflow.com/questions/24193174/reset-color-cycle-in-matplotlib
    plt.gca().set_prop_cycle(None)
    plt.semilogx(freqs[1:num_bins], final_freq_response[1:num_bins, ::1])
    plt.ylabel('dB')
    plt.xlabel('FFT Bin')
    plt.title(
        'NP: Initial (dotted) vs. Adapted at 1kHz (solid) Frequency Response'
    )
    plt.ylim(-100, -15)
    plt.savefig('/tmp/whole_carfac_response.png')

    initial_resps = [
        find_peak_response(freqs, initial_freq_response[:, i])
        for i in range(71)
    ]
    final_resps = [
        find_peak_response(freqs, final_freq_response[:, i]) for i in range(71)
    ]
    initial_resps = np.asarray(initial_resps)
    final_resps = np.asarray(final_resps)

    plt.figure()
    plt.plot(initial_resps[:, 0], ':')
    plt.plot(final_resps[:, 0])
    plt.xlabel('Ear Channel #')
    plt.ylabel('dB')
    plt.title('NP: Initial (dotted) vs. Adapted (solid) Center Frequencies')
    plt.savefig('/tmp/whole_carfac_CF.png')

    plt.figure()
    plt.plot(initial_resps[:, 1], ':')
    plt.plot(final_resps[:, 1])
    plt.xlabel('Ear Channel #')
    plt.title('NP: Initial (dotted) vs. Adapted (solid) Peak Gain')
    plt.savefig('/tmp/whole_carfac_peak_gain.png')

    # Test for change in peak gain after adaptation.
    def find_closest_channel(cfs: List[float], desired: float) -> np.ndarray:
      return np.argmin((np.asarray(cfs) - desired) ** 2)

    results = {}
    if ihc_style == 'one_cap':
      results = {  # The Matlab test prints this data block:
          125: [64, 119.007, 0.264],
          250: [58, 239.791, 0.986],
          500: [49, 514.613, 7.309],
          1000: [38, 1099.436, 31.644],
          2000: [28, 2038.875, 27.214],
          4000: [16, 4058.882, 13.823],
          8000: [2, 8289.883, 3.565],
      }
    else:
      # The matlab test, with a change to two cap, prints this data block:
      results = {
          125: [64, 119.007, 0.258],
          250: [58, 239.791, 0.963],
          500: [49, 514.613, 7.224],
          1000: [38, 1099.436, 31.373],
          2000: [28, 2038.875, 26.244],
          4000: [16, 4058.882, 12.726],
          8000: [2, 8289.883, 3.212],
      }
    # Print out these results first, before possible test failure.
    for desired_cf in results:
      c = find_closest_channel([f[0] for f in final_resps], desired_cf)
      expected_c, expected_cf, expected_change = results[desired_cf]
      cf = initial_resps[expected_c][0]
      diff_db = initial_resps[expected_c][1] - final_resps[expected_c][1]
      print(
          f'Desired CF={desired_cf}Hz: Expected to find channel'
          f' {expected_c} with a CF of {expected_cf}Hz and gain change of'
          f' {expected_change}dB.'
      )
      print(
          f'                        Instead found channel {int(c)} with a CF '
          f'of {cf}Hz and gain change of {diff_db}dB.'
      )
    # Print a data block, just like the matlab does, for copy-paste purposes.
    for desired_cf in results:
      c = find_closest_channel([f[0] for f in final_resps], desired_cf)
      expected_c, expected_cf, expected_change = results[desired_cf]
      cf = initial_resps[expected_c][0]
      diff_db = initial_resps[expected_c][1] - final_resps[expected_c][1]
      print(f'{desired_cf}: [{int(c)}, {cf:.3f}, {diff_db:.3f}],')

    for cf in results:
      c = find_closest_channel([f[0] for f in final_resps], cf)
      expected_c, expected_cf, expected_change = results[cf]
      cf = initial_resps[expected_c][0]
      diff_db = initial_resps[expected_c][1] - final_resps[expected_c][1]
      self.assertEqual(c, expected_c)
      self.assertAlmostEqual(cf, expected_cf, delta=expected_cf / 10000.0)
      self.assertAlmostEqual(
          diff_db, expected_change, delta=0.01
      )  # dB change diff.

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
        cfp, impulse, open_loop=True, linear_car=True
    )
    plt.imshow(bm_initial[:80, :])
    plt.title('Impulse response with direct connections between stages')

    cfp = carfac.carfac_init(cfp)  # Clear state to zero between runs.
    cfp.ears[0].car_coeffs.use_delay_buffer = True
    _, cfp, bm_delayed, _, _ = carfac.run_segment(
        cfp, impulse, open_loop=True, linear_car=True
    )
    plt.figure()
    plt.imshow(bm_delayed[:80, :])
    plt.title('Impulse response with delay buffer (one extra delay per stage)')

    max_max_rel_error = 0
    for ch in np.arange(cfp.n_ch):
      impresp = bm_initial[: -ch - 1, ch]
      delayed = bm_delayed[ch:-1, ch]
      max_abs = np.amax(np.abs(impresp))
      max_abs_error = np.amax(np.abs(impresp - delayed))
      max_rel = max_abs_error / max_abs
      max_max_rel_error = np.maximum(max_max_rel_error, max_rel)
    print(max_max_rel_error)
    self.assertLess(max_max_rel_error, 4e-4)  # More tolerance than Matlab. Why?

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
      impresp = bm_initial[: -ch - 1, ch]
      delayed = bm_delayed[ch:-1, ch]
      max_abs = np.amax(np.abs(impresp))
      max_abs_error = np.amax(np.abs(impresp - delayed))
      max_rel = max_abs_error / max_abs
      max_max_rel_error = np.maximum(max_max_rel_error, max_rel)
    print(max_max_rel_error)
    self.assertLess(max_max_rel_error, 0.025)

  def test_ohc_health(self):
    # Test: Verify reduce gain with reduced OHC health.
    # Set the random seed.
    np.random.seed(seed=1)

    fs = 22050.0
    t = np.arange(0, 1, 1 / fs)  # A second of noise.
    amplitude = 1e-4  # -80 dBFS, around 20 or 30 dB SPL
    noise = amplitude * np.random.randn(len(t))
    cfp = carfac.design_carfac(fs=fs)
    cfp = carfac.carfac_init(cfp)
    # Run the healthy case with low-level noise.
    _, cfp, bm_baseline, _, _ = carfac.run_segment(cfp, noise)

    # Make the OHCs less healthy over the basal half.
    n_ch = cfp.n_ch
    half_ch = int(np.floor(n_ch / 2))
    cfp.ears[0].car_coeffs.ohc_health[range(half_ch)] *= 0.5  # Degrade.

    cfp = carfac.carfac_init(cfp)  # Clear state to zero between runs.
    _, cfp, bm_degraded, _, _ = carfac.run_segment(cfp, noise)

    # Compare rms outputs per channel.
    rms_baseline = np.mean(bm_baseline**2, axis=0) ** 0.5
    rms_degraded = np.mean(bm_degraded**2, axis=0) ** 0.5
    tf_ratio = rms_degraded / rms_baseline

    plt.clf()
    plt.plot(tf_ratio)
    plt.title('TF ratio with degraded OHC in basal half')

    # Expect tf_ratio low in early channels, closer to 1 later.
    for ch in np.arange(9, half_ch):
      self.assertLess(tf_ratio[ch], 0.11)  # 0.1 usually works, depends on seed.
    for ch in np.arange(half_ch + 5, n_ch - 2):
      self.assertGreater(tf_ratio[ch], 0.35)

  def test_multiaural_carfac(self):
    """Test multiaural functionality with 2 ears.

    Tests that in binaural carfac, providing identical noise to both ears
    gives idental nap output at end.
    """
    # for now only test 2 ears.
    np.random.seed(seed=1)

    fs = 22050.0
    t = np.arange(0, 1, 1 / fs)  # A second of noise.
    amplitude = 1e-3  # -70 dBFS, around 30 or 40 dB SPL
    noise = amplitude * np.random.randn(len(t))
    two_chan_noise = np.zeros((len(t), 2))
    two_chan_noise[:, 0] = noise
    two_chan_noise[:, 1] = noise
    cfp = carfac.design_carfac(fs=fs, n_ears=2, ihc_style='one_cap')
    cfp = carfac.carfac_init(cfp)
    naps, _, _, _, _ = carfac.run_segment(cfp, two_chan_noise)
    max_abs_diff = np.amax(np.abs(naps[:, :, 0] - naps[:, :, 1]))
    self.assertLess(max_abs_diff, 1e-5)

  def test_multiaural_carfac_with_silent_channel(self):
    """Test multiaural functionality with 2 ears.

    Runs a 50ms sample of a pair of C Major chords, and tests a binaural carfac
    with 1 silent ear against a simple monoaural carfac with only the chords as
    input.

    Tests that:
    1. The ratio of BM Movement is within an expected range [1, 1.25]
    2. Tests the precise ratio between the two, taken as golden data from the
    matlab
    """
    # for now only test 2 ears.
    fs = 22050.0
    t = np.arange(0, 0.05 - 1 / fs, 1 / fs)  # 50ms of times.
    t_prime = t.reshape(1, len(t))
    amplitude = 1e-3  # -70 dBFS, around 30 or 40 dB SPL

    # c major chord at 4th octave . C-E-G, 523.25-659.25-783.99
    # and then a few octaves lower, at 32.7 41.2 and 49.
    freqs = np.asarray(
        [523.25, 659.25, 783.99, 32.7, 41.2, 49], dtype=np.float64
    )
    freqs = freqs.reshape(len(freqs), 1)
    c_major_chord = amplitude * np.sum(
        np.sin(2 * np.pi * np.matmul(freqs, t_prime)), 0
    )

    two_chan_noise = np.zeros((len(t), 2))
    two_chan_noise[:, 0] = c_major_chord
    # Leave the audio in channel 1 as silence.
    cfp = carfac.design_carfac(fs=fs, n_ears=2, ihc_style='one_cap')
    cfp = carfac.carfac_init(cfp)
    mono_cfp = carfac.design_carfac(fs=fs, n_ears=1, ihc_style='one_cap')
    mono_cfp = carfac.carfac_init(mono_cfp)

    _, _, bm_binaural, _, _ = carfac.run_segment(cfp, two_chan_noise)
    _, _, bm_monoaural, _, _ = carfac.run_segment(mono_cfp, c_major_chord)

    bm_mono_ear = bm_monoaural[:, :, 0]
    rms_bm_mono = np.sqrt(np.mean(bm_mono_ear**2, axis=0))

    bm_good_ear = bm_binaural[:, :, 0]
    rms_bm_binaural_good_ear = np.sqrt(np.mean(bm_good_ear**2, axis=0))

    tf_ratio = rms_bm_binaural_good_ear / rms_bm_mono
    # this data comes directly from the same test that executes in Matlab.
    expected_tf_ratio = [
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0000,
        1.0001,
        1.0001,
        1.0001,
        1.0002,
        1.0004,
        1.0007,
        1.0018,
        1.0050,
        1.0133,
        1.0290,
        1.0463,
        1.0562,
        1.0552,
        1.0505,
        1.0497,
        1.0417,
        1.0426,
        1.0417,
        1.0320,
        1.0110,
        1.0093,
        1.0124,
        1.0065,
        1.0132,
        1.0379,
        1.0530,
        1.0503,
        1.0477,
        1.0556,
        1.0659,
        1.0739,
        1.0745,
        1.0762,
        1.0597,
        1.0200,
        1.0151,
        1.0138,
        1.0129,
        1.0182,
    ]
    diff_ratio = np.abs(tf_ratio - expected_tf_ratio)
    for ch in np.arange(len(diff_ratio)):
      self.assertAlmostEqual(
          tf_ratio[ch],
          expected_tf_ratio[ch],
          places=3,
          msg='Failed at channel %d' % ch,
      )
    self.assertTrue(np.all(tf_ratio >= 1))
    self.assertTrue(np.all(tf_ratio <= 1.25))


if __name__ == '__main__':
  absltest.main()
