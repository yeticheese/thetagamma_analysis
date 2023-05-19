import numpy as np
import scipy


def get_rem_states(rem_states, sample_rate):
    rem_state_indices = np.where(rem_states == 5)[1]
    rem_state_changes = np.diff(rem_state_indices)
    split_indices = np.where(rem_state_changes != 1)[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(rem_state_indices)]))
    consecutive_rem_states = np.empty((len(split_indices)-1, 2))
    for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
        start = rem_state_indices[start] * sample_rate
        end = rem_state_indices[end - 1] * sample_rate
        consecutive_rem_states[i] = np.array([start, end])
    consecutive_rem_states = np.array(consecutive_rem_states)
    return consecutive_rem_states


def morlet_wt(x, sample_rate=2500, freq_range=(1, 200), tcenter=4, n=5, zscore=True, mode='amplitude'):
    global convuP
    tcenter = np.arange(np.negative(tcenter), tcenter, 1 / sample_rate)
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


def tg_split(mask_freq, freq_range=(5, 12)):
    lower = np.min(freq_range)
    upper = np.max(freq_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = [sub_mask_index]
    theta = [mask_index]
    supra = [supra_mask_index]

    return sub, theta, supra
