import warnings
warnings.simplefilter('ignore')
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import mir_eval
import matplotlib.pyplot as plt
import scipy.stats


def periodicity_estimation_plots(oenv, sr, hop_length, ref_beats=None):
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                          hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                               hop_length=hop_length)[0]


    fig, ax = plt.subplots(nrows=4, figsize=(14, 14))
    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    ax[0].plot(times, oenv)
    ax[0].set_title('Spectral flux',fontsize=15)
    ax[0].label_outer()
    ax[0].set(xlim=[0, len(oenv)/fps]);
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[1])
    ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                label='Estimated tempo={:g}'.format(tempo))
    ax[1].legend(loc='upper right')
    ax[1].set_title('Tempogram',fontsize=15)
    x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
                    num=tempogram.shape[0])
    ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    ax[2].set(xlabel='Lag (seconds)')
    ax[2].legend(frameon=True)
    freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
                 label='Mean local autocorrelation', base=2)
    ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                 label='Global autocorrelation', base=2)
    ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,
                label='Estimated tempo={:g}'.format(tempo))
    
    if ref_beats is not None:
        gt_tempo = 60./np.median(np.diff(ref_beats))
        ax[3].axvline(gt_tempo, color='red', linestyle='--', alpha=.8,
                label='Tempo derived from beat annotations={:g}'.format(gt_tempo))

    ax[3].legend(frameon=True)
    ax[3].legend(loc='upper right')
    ax[3].set(xlabel='BPM')
    ax[3].grid(True)
    return tempo

def beat_track_dp(oenv, tempo, fps, sr, hop_length, tightness=100, alpha=0.5, ref_beats=None):

    period = (fps * 60./tempo)
    localscore = librosa.beat.__beat_local_score(oenv, period)
    
    backlink = np.zeros_like(localscore, dtype=int)
    cumulative_score = np.zeros_like(localscore)

    # Search range for previous beat
    window = np.arange(-2 * period, -np.round(period / 2) + 1, dtype=int)

    txwt = -tightness * (np.log(-window / period) ** 2)

    # Are we on the first beat?
    first_beat = True
    for i, score_i in enumerate(localscore):

        # Are we reaching back before time 0?
        z_pad = np.maximum(0, min(-window[0], len(window)))

        # Search over all possible predecessors
        candidates = txwt.copy()
        candidates[z_pad:] = candidates[z_pad:] + cumulative_score[window[z_pad:]]

        # Find the best preceding beat
        beat_location = np.argmax(candidates)

        # Add the local score
        cumulative_score[i] = (1-alpha)*score_i + alpha*candidates[beat_location]

        # Special case the first onset.  Stop if the localscore is small
        if first_beat and score_i < 0.01 * localscore.max():
            backlink[i] = -1
        else:
            backlink[i] = window[beat_location]
            first_beat = False

        # Update the time range
        window = window + 1

    beats = [librosa.beat.__last_beat(cumulative_score)]

    # Reconstruct the beat path from backlinks
    while backlink[beats[-1]] >= 0:
        beats.append(backlink[beats[-1]])

    # Put the beats in ascending order
    # Convert into an array of frame numbers
    beats = np.array(beats[::-1], dtype=int)

    # Discard spurious trailing beats
    beats = librosa.beat.__trim_beats(oenv, beats, trim=True)
    
    # Convert beat times seconds
    beats = librosa.frames_to_time(beats, hop_length=hop_length, sr=sr)
    
    return beats, cumulative_score

# again we'll make a little helper function for plotting,
# we can pass the annotate beats (ref_beats) just for plotting purposes
def dp_and_plot(oenv, tempo, fps, sr, hop_length, tightness, alpha, ref_beats=None):
    
    est_beats, cumulative_score = beat_track_dp(oenv, tempo, fps, 
                                                sr, hop_length, tightness, alpha, ref_beats)
    fig, ax = plt.subplots(nrows=2, figsize=(14, 6))
    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    ax[0].plot(times, oenv, label='Spectral flux')
    ax[0].set_title('Spectral flux',fontsize=15)
    ax[0].label_outer()
    if ref_beats is not None:
        ax[0].vlines(ref_beats, 0, 1.1*oenv.max(), label='Annotated Beats', 
                     color='r', linestyle=':', linewidth=2)

    ax[0].set(xlim=[0, len(oenv)/fps])
    ax[0].legend(loc='upper right')

    ax[1].plot(times, cumulative_score, color='orange', label='Cumultative score')
    ax[1].set_title('Cumulative score (alpha:'+str(alpha)+')',fontsize=15)
    ax[1].label_outer()
    ax[1].set(xlim=[0, len(oenv)/fps])
    ax[1].vlines(est_beats, 0, 1.1*cumulative_score.max(), label='Estimated beats', 
                 color='green', linestyle=':', linewidth=2)
    ax[1].legend(loc='upper right')
    ax[1].set(xlabel = 'Time')
    
    return est_beats


#file to be tested
filename = './easy_example'

#fractions of a second
fps = 100
#sample rate
sr = 44100
n_fft = 2048
hop_length = int(librosa.time_to_samples(1./fps, sr=sr))
n_mels = 80
fmin = 27.5
fmax = 17000.
lag = 2
max_size = 3

# read audio and annotations
y, sr = librosa.load(filename+'.flac', sr = sr)
ref_beats = np.loadtxt(filename+'.beats')
ref_beats = ref_beats[:,0]

# make the mel spectrogram
S = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)

S_bl = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=400, n_mels=40)

# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(14,6))

# librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax[0], color="blue")

# ax[0].set_title('Easy Example: audio waveform', fontsize=15)
# ax[0].label_outer()

# librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, ax=ax[1])

# ax[1].set_title('Mel Spectrogram', fontsize=15)
# ax[1].label_outer()

spectral_flux = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sr,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)


# bl_spectral_flux = librosa.onset.onset_strength(S=librosa.power_to_db(S_bl, ref=np.max),
#                                       sr=sr,
#                                       hop_length=hop_length,
#                                       lag=lag, max_size=max_size)


# frame_time = librosa.frames_to_time(np.arange(len(spectral_flux)),
#                                     sr=sr,
#                                     hop_length=hop_length)

# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(14,6))

# ax[0].plot(frame_time, spectral_flux, label='Spectral flux')
# ax[0].set_title('Spectral flux', fontsize=15)

# ax[1].plot(frame_time, bl_spectral_flux, label='Band-limited spectral flux')
# ax[1].set_title('Band-limited spectral flux', fontsize=15)
# ax[1].set(xlabel='Time')
# ax[1].set(xlim=[0, len(y)/sr])
# ax[0].label_outer()

tempo = periodicity_estimation_plots(oenv=spectral_flux, sr=sr, hop_length=hop_length)

dp_and_plot(oenv=spectral_flux, tempo=tempo, fps=fps, sr=sr,hop_length=hop_length, tightness=100, alpha=0.5, ref_beats = ref_beats)

plt.show()