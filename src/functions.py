import numpy as np
import emd.sift as sift
import emd.spectra as spectra
import scipy


def get_rem_states(states, sample_rate):
    states = np.squeeze(states)
    rem_state_indices = np.where(states == 5)[0]
    rem_state_changes = np.diff(rem_state_indices)
    split_indices = np.where(rem_state_changes != 1)[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(rem_state_indices)]))
    consecutive_rem_states = np.empty((len(split_indices) - 1, 2))
    for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
        start = rem_state_indices[start] * int(sample_rate)
        end = rem_state_indices[end - 1] * int(sample_rate)
        consecutive_rem_states[i] = np.array([start, end])
    consecutive_rem_states = np.array(consecutive_rem_states)
    null_states_mask = np.squeeze(np.diff(consecutive_rem_states) > 0)
    consecutive_rem_states = consecutive_rem_states[null_states_mask]
    return consecutive_rem_states


def morlet_wt(x, s_rate=2500, freq_range=(1, 200), tcenter=4, n=5):
    freq = np.arange(np.min(freq_range), np.max(freq_range), 1)
    wavelet_transform = np.zeros((len(freq), len(x)), dtype=complex)
    for i, freq in enumerate(freq):
        h = (n * (2 * np.log(2)) ** 0.5) / (np.pi * freq)
        tcenter = np.arange(len(x)) / s_rate - 0.5 * len(x) / s_rate
        wavelet = np.exp(1j * 2 * np.pi * freq * tcenter) * np.exp((-4 * np.log(2) * tcenter ** 2) / h ** 2)
        wavelet_transform[i, :] = np.convolve(x.T, wavelet, mode='same')
    return wavelet_transform


def tg_split(mask_freq, theta_range=(5, 12)):
    lower = np.min(theta_range)
    upper = np.max(theta_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = sub_mask_index
    theta = mask_index
    supra = supra_mask_index

    return sub, theta, supra


def zero_cross(x):
    decay = np.logical_and((x > 0)[1:], ~(x > 0)[:-1]).nonzero()[0]
    rise = np.logical_and((x <= 0)[1:], ~(x <= 0)[:-1]).nonzero()[0]
    zero_xs = np.sort(np.append(rise, decay))
    return zero_xs


def extrema(x):
    zero_xs = zero_cross(x)
    peaks = np.empty((0,)).astype(int)
    troughs = np.empty((0,)).astype(int)
    for t1, t2 in zip(zero_xs, zero_xs[1:]):
        extrema0 = np.argmax(np.abs(x[t1:t2])).astype(int) + t1
        if bool(x[extrema0] > 0):
            peaks = np.append(peaks, extrema0)
        else:
            troughs = np.append(troughs, extrema0)
    return zero_xs, troughs, peaks


def get_cycles_data(x, rem_states, sample_rate, theta_range=(5, 12)):
    consecutive_rem_states = get_rem_states(rem_states, sample_rate)
    rem_imf = []
    rem_mask_freq = []
    instantaneous_phase = []
    instantaneous_freq = []
    instantaneous_amp = []
    sub_theta_sig = np.empty((0,))
    theta_peak_sig = np.empty((0,))
    cycles = np.empty((0, 5)).astype(int)
    rem_dict = {}
    sub_dict = rem_dict

    for i, rem in enumerate(consecutive_rem_states, start=1):
        sub_dict.setdefault(f'REM {i}', {})
        start = int(rem[0])
        end = int(rem[1])
        signal = x[start:end]
        imf, mask_freq = sift.iterated_mask_sift(signal,
                                                 mask_0='zc',
                                                 sample_rate=sample_rate,
                                                 ret_mask_freq=True)
        IP, IF, IA = spectra.frequency_transform(imf, sample_rate, 'nht')
        sub_theta, theta, _ = tg_split(mask_freq, theta_range)

        rem_imf.append(imf)
        rem_mask_freq.append(mask_freq)
        instantaneous_phase.append(IP)
        instantaneous_freq.append(IF)
        instantaneous_amp.append(IA)

        theta_sig = np.sum(imf.T[theta], axis=0)
        sub_theta_sig = np.append(sub_theta_sig, np.sum(imf.T[sub_theta], axis=0))

        zero_x, trough, peak = extrema(np.sum(imf.T[theta], axis=0))

        zero_x = np.vstack((zero_x[:-2:2], zero_x[1:-1:2], zero_x[2::2])).T

        size_adjust = np.min([trough.shape[0], zero_x.shape[0], peak.shape[0]])
        zero_x = zero_x[:size_adjust]
        cycle = np.empty((size_adjust, 5)).astype(int)
        cycle[:, [0, 2, 4]] = zero_x
        if trough[0] < peak[0]:
            cycle[:, 1] = trough[:zero_x.shape[0]]
            cycle[:, 3] = peak[:zero_x.shape[0]]
        else:
            cycle[:, 3] = trough[:zero_x.shape[0]]
            cycle[:, 1] = peak[:zero_x.shape[0]]

        broken_cycle = cycle[~np.all(np.diff(cycle, axis=1) > 0, axis=1)]
        broken_cycle_mask = np.diff(broken_cycle, axis=1) > 0

        adjust_condition = np.all(np.all(broken_cycle_mask[1:] == [True, False, False, True],
                                         axis=0) == True)
        adjust_loc = np.where(np.all(np.diff(cycle, axis=1) > 0, axis=1) == False)[0][1:-1]

        fixed_cycle = broken_cycle[1:-1]
        if adjust_condition:
            fixed_cycle[:, 1] = cycle[adjust_loc - 1, 1]
            fixed_cycle[:, 3] = cycle[adjust_loc + 1, 3]
        else:
            fixed_cycle[:, 3] = cycle[adjust_loc - 1, 3]
            fixed_cycle[:, 1] = cycle[adjust_loc + 1, 1]

        cycle = cycle[np.all(np.diff(cycle, axis=1) > 0, axis=1)]
        cycle = np.vstack((cycle, fixed_cycle))
        if trough[0] < peak[0]:
            cycle = np.hstack((cycle[:-1, 1:-1], cycle[1:, :2]))
        else:
            cycle = np.hstack((cycle[:-1, 3].reshape((-1, 1)), cycle[1:, :-1]))

        theta_peak_sig = np.append(theta_peak_sig, theta_sig[cycle[:, 2]])
        cycles = np.vstack((cycles, cycle + start))

    min_peak_amp = 2 * sub_theta_sig.std()
    peak_mask = theta_peak_sig > min_peak_amp
    upper_diff = np.floor(1000 / np.min(theta_range))
    lower_diff = np.floor(1000 / np.max(theta_range))
    diff_mask = np.logical_and(np.diff(cycles[:, [0, -1]], axis=1) * (1000 / sample_rate) > lower_diff,
                               np.diff(cycles[:, [0, -1]], axis=1) * (1000 / sample_rate) <= upper_diff)

    extrema_mask = np.logical_and(np.squeeze(diff_mask), peak_mask)

    cycles = cycles[extrema_mask]

    for j, rem in enumerate(rem_dict.values()):
        rem['start-end'] = consecutive_rem_states[j]
        rem['IMFs'] = rem_imf[j]
        rem['IMF_Frequencies'] = rem_mask_freq[j]
        rem['Instantaneous Phases'] = instantaneous_phase[j]
        rem['Instantaneous Frequencies'] = instantaneous_freq[j]
        rem['Instantaneous Amplitudes'] = instantaneous_amp[j]
        cycles_mask = (cycles > consecutive_rem_states[j, 0]) & (cycles < consecutive_rem_states[j, 1])
        cycles_mask = np.all(cycles_mask == True, axis=1)
        rem_cycles = cycles[cycles_mask]
        rem['Cycles'] = rem_cycles
    return rem_dict


def bin_tf_to_fpp(x, power, bin_count):
    if x.ndim == 1:  # Handle the case when x is of size (2)
        bin_ranges = np.arange(x[0], x[1], 1)
        fpp = scipy.stats.binned_statistic(bin_ranges, power[:, x[0]:x[1]], 'mean', bins=bin_count)[0]
        fpp = np.expand_dims(fpp, axis=0)  # Add an extra dimension to match the desired output shape
    elif x.ndim == 2:  # Handle the case when x is of size (n, 2)
        fpp = []
        for i in range(x.shape[0]):
            bin_ranges = np.arange(x[i, 0], x[i, 1], 1)
            fpp_row = scipy.stats.binned_statistic(bin_ranges, power[:, x[i, 0]:x[i, 1]], 'mean', bins=bin_count)[0]
            fpp.append(fpp_row)
        fpp = np.array(fpp)
    else:
        raise ValueError("Invalid size for x")

    return fpp


def calculate_cog(frequencies, angles, amplitudes):
    angles = np.deg2rad(angles)
    if amplitudes.ndim == 2:
        numerator = np.sum(frequencies * np.sum(amplitudes, axis=1))
        denominator = np.sum(amplitudes)
        cog_f = numerator / denominator
        cog_ph = np.rad2deg(pg.circ_mean(angles, w=np.sum(amplitudes, axis=0)))
        cog = np.array([cog_f, cog_ph])
    elif amplitudes.ndim == 3:
        cog = []
        numerator = np.sum(amplitudes, axis=2)
        denominator = np.sum(amplitudes, axis=(1, 2))
        circ_weights = np.sum(amplitudes, axis=1)
        for i in range(amplitudes.shape[0]):
            cog_f = np.sum(frequencies * numerator[i]) / denominator[i]
            cog_ph = np.rad2deg(pg.circ_mean(angles, w=circ_weights[i]))
            cog_cycle = [cog_f, cog_ph]
            cog.append(cog_cycle)
        cog = np.array(cog)
    return cog
