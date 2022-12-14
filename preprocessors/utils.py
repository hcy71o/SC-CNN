import numpy as np 

def get_alignment(tier, sampling_rate, hop_length):
    sil_phones = ['sil', 'sp', 'spn', '']
    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(round(e*sampling_rate/hop_length)-round(s*sampling_rate/hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time


def is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return x <= lower or x >= upper

def remove_outlier(values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]    

def average_by_duration(x, durs):
    length = sum(durs)
    durs_cum = np.cumsum(np.pad(durs, (1, 0), mode='constant'))

    # calculate charactor f0/energy
    if len(x.shape) == 2:
        x_char = np.zeros((durs.shape[0], x.shape[1]), dtype=np.float32)
    else:
        x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(length), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values, axis=0) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)


 

