import numpy as np
import scipy
import emd.sift as sift
import emd.spectra as spectra


def get_rem_states(rem_states, s_rate):
    rem_state_indices = np.where(rem_states == 5)[1]
    rem_state_changes = np.diff(rem_state_indices)
    split_indices = np.where(rem_state_changes != 1)[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(rem_state_indices)]))
    consecutive_rem_states = np.empty((len(split_indices) - 1, 2))
    for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
        start = rem_state_indices[start] * s_rate
        end = rem_state_indices[end - 1] * s_rate
        consecutive_rem_states[i] = np.array([start, end])
    consecutive_rem_states = np.array(consecutive_rem_states)
    return consecutive_rem_states


def morlet_wt(x, s_rate=2500, freq_range=(1, 200), tcenter=4, n=5, zscore=True, mode='amplitude'):
    global convuP
    tcenter = np.arange(np.negative(tcenter), tcenter, 1 / s_rate)
    nSignal = len(x)

    freq_vec = np.arange(np.min(freq_range), np.max(freq_range), 1)

    tf = np.empty((len(freq_vec), len(x)))

    for i, fi in enumerate(freq_vec):
        h = (n * (2 * np.log(2)) ** 0.5) / (np.pi * fi)
        cmorl = np.exp(1j * 2 * np.pi * fi * tcenter) * np.exp((-4 * np.log(2) * tcenter ** 2) / h ** 2)
        nKern = len(cmorl)
        halfKern = int(np.floor(nKern / 2))
        nConvu = nSignal + nKern - 1
        cmorlX = scipy.fft.fft(cmorl, nConvu)
        cmorlX = cmorlX / max(cmorlX)
        lfpX = scipy.fft.fft(x, nConvu)
        convuX = cmorlX * lfpX
        convu = scipy.fft.ifft(convuX)
        start = halfKern
        end = len(convu) - start
        convuFix = convu[start:end + 1]
        if mode == 'amplitude':
            convuP = np.abs(convuFix)
        elif mode == 'power':
            convuP = np.abs(convuFix) ** 2

        tf[i:] = convuP
    if zscore:
        tfz = scipy.stats.zscore(tf, axis=0)
        return tfz
    else:
        return tf


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


def get_cycles_data(x, rem_states, s_rate, theta_range=(5, 12)):
    consecutive_rem_states = get_rem_states(rem_states, s_rate)
    rem_imf = []
    rem_mask_freq = []
    instantaneous_phase = []
    instantaneous_freq = []
    instantaneous_amp = []
    sub_theta_sig = np.empty((0,))
    zero_xs = np.empty((0, 2)).astype(int)
    troughs = np.empty((0, 2)).astype(int)
    peaks = np.empty((0, 1)).astype(int)
    theta_peak_sig = np.empty((0,))

    rem_dict = {}
    sub_dict = rem_dict

    for i, rem in enumerate(consecutive_rem_states):
        sub_dict.setdefault(f'REM {i + 1}', {})
        start = int(rem[0])
        end = int(rem[1])
        signal = x[start:end]
        imf, mask_freq = sift.iterated_mask_sift(signal,
                                                 mask_0='zc',
                                                 sample_rate=s_rate,
                                                 ret_mask_freq=True)
        IP, IF, IA = spectra.frequency_transform(imf, s_rate, 'nht')
        sub_theta, theta, _ = tg_split(mask_freq, theta_range)

        rem_imf.append(imf)
        rem_mask_freq.append(mask_freq)
        instantaneous_phase.append(IP)
        instantaneous_freq.append(IF)
        instantaneous_amp.append(IA)

        theta_sig = np.sum(imf.T[theta], axis=0)
        sub_theta_sig = np.append(sub_theta_sig, np.sum(imf.T[sub_theta], axis=0))

        zero_x, trough, peak = extrema(np.sum(imf.T[theta], axis=0))
        zero_x = zero_x[(zero_x > trough[0]) & (zero_x < trough[-1])]
        peak = peak[(peak > zero_x[0]) & (peak < zero_x[-1])]
        zero_x = zero_x.reshape(-1, 2)
        trough = np.vstack((trough[:-1], trough[1:])).T

        zero_xs = np.vstack((zero_xs, zero_x + start))
        troughs = np.vstack((troughs, trough + start))
        peaks = np.append(peaks, peak + start)
        theta_peak_sig = np.append(theta_peak_sig, theta_sig[peak])

    min_peak_amp = 2*sub_theta_sig.std()
    peak_mask = theta_peak_sig > min_peak_amp

    upper_diff = np.floor(1000 / np.min(theta_range))
    lower_diff = np.floor(1000 / np.max(theta_range))
    diff_mask = np.logical_and(np.diff(troughs)*(1000/s_rate) > lower_diff,
                               np.diff(troughs)*(1000/s_rate) <= upper_diff).reshape(-1,)

    extrema_mask = np.logical_and(diff_mask, peak_mask)
    troughs = troughs[extrema_mask]
    peaks = peaks[extrema_mask]
    zero_xs = zero_xs[extrema_mask]

    for j, rem in enumerate(rem_dict.values()):
        rem['start-end'] = consecutive_rem_states[j]
        rem['IMFs'] = rem_imf[j]
        rem['IMF_Frequencies'] = rem_mask_freq[j]
        rem['Instantaneous Phases'] = instantaneous_phase[j]
        rem['Instantaneous Frequencies'] = instantaneous_freq[j]
        rem['Instantaneous Amplitudes'] = instantaneous_amp[j]
        rem['Zero Crossings'] = zero_xs[
            (zero_xs >= consecutive_rem_states[j, 0]) & (zero_xs <= consecutive_rem_states[j, 1])]
        rem['Troughs'] = troughs[(troughs >= consecutive_rem_states[j, 0]) & (troughs <= consecutive_rem_states[j, 1])]
        rem['Peaks'] = peaks[(peaks >= consecutive_rem_states[j, 0]) & (peaks <= consecutive_rem_states[j, 1])]
    return rem_dict
