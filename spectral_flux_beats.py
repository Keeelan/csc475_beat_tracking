import os
import madmom
import librosa
import soundfile as sf
from madmom.features.beats import BeatTrackingProcessor, RNNBeatProcessor
import numpy as np
import matplotlib.pyplot as plt
import mir_eval
import csv


def save_f_measure_scores(f_measure_data):
    # Create directory if it doesn't exist
    #os.makedirs(os.path.dirname("f_measure_results.csv"), exist_ok=True)
    output_file = "f_measure_results.csv"

    # Open the CSV file for writing
    with open(output_file, "w") as csvfile:
        writer = csv.writer(csvfile)

        # Write header row
        writer.writerow(["Genre", "Average F-measure"])

        # Write genre-wise and overall average scores
        for genre, score in f_measure_data.items():
            writer.writerow([genre, score])

        writer.writerow(["Overall", f_measure_data["overall"]])

    print(f"F-measure scores saved to {output_file}")


def calculate_beats(audio_file_path):
    
    audio, sample_rate = librosa.load(audio_file_path, sr=None)  # Assuming you have librosa installed
        
    fps = 100
    n_fft = 2048
    hop_length = int(librosa.time_to_samples(1./fps, sr=sample_rate))
    n_mels = 80
    fmin = 27.5
    fmax = 17000.
    lag = 2
    max_size = 3
    tightness = 100
    alpha = 0.5

    S = librosa.feature.melspectrogram(y = audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)

    spectral_flux = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sample_rate,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)

    predicted_tempo = librosa.beat.tempo(onset_envelope=spectral_flux, sr=sample_rate,
                               hop_length=hop_length)[0]

    period = (fps * 60./predicted_tempo)
    localscore = librosa.beat.__beat_local_score(spectral_flux, period)
    
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
    beats = librosa.beat.__trim_beats(spectral_flux, beats, trim=True)
    
    # Convert beat times seconds
    beats = librosa.frames_to_time(beats, hop_length=hop_length, sr=sample_rate)
    
    return beats      

def load_beats(file_path):
    with open(file_path, 'r') as file:
        beats = [float(line.split()[0]) for line in file]
    return beats

def extract_genre_and_file_number(file_name):
    # Splitting the file name by dots
    file_parts = os.path.splitext(file_name)[0].split('.')

    # Extracting genre and file number
    if len(file_parts) == 2:
        genre, file_number = file_parts
        return genre, file_number.zfill(5)
    else:
        return None, None
    
def evaluate_beat_tracking_algorithm(algorithm_beats_folder, ground_truth_beats_folder):
    genres = os.listdir(algorithm_beats_folder)

    f_measure_data = {}
    
    total_f_measure_sum = 0

    total_num_files = 0

    for genre in genres:
        genre_path = os.path.join(algorithm_beats_folder, genre)
        if not os.path.isdir(genre_path):
            continue

        genre_files = os.listdir(genre_path)

        num_files = 0

        genre_f_measure_sums = {}

        
        for file_name in genre_files:
            genre_part, file_number = extract_genre_and_file_number(file_name)
            if genre_part is None or file_number is None:
                print(f"Skipping invalid file name: {file_name}")
                continue

            algorithm_audio_path = os.path.join(genre_path, file_name)
            ground_truth_beats_path = os.path.join(ground_truth_beats_folder, f"gtzan_{genre_part}_{file_number}.beats")

            
            if not os.path.isfile(ground_truth_beats_path):
                print(f"Ground truth beats not found for {file_name}")
                continue

        

            algorithm_beats = calculate_beats(algorithm_audio_path)
            ground_truth_beats = load_beats(ground_truth_beats_path)

                        
            algorithm_beats_array = np.array(algorithm_beats)
            ground_truth_beats_array = np.array(ground_truth_beats)

            reference_beats = mir_eval.beat.trim_beats(ground_truth_beats_array)
            estimated_beats = np.array(mir_eval.beat.trim_beats(algorithm_beats_array))

            f_measure = mir_eval.beat.f_measure(reference_beats, estimated_beats)
            print(f"F-measure for {file_name}: {f_measure}")

            genre_f_measure_sums[genre] = genre_f_measure_sums.get(genre, []) + [f_measure]
            

            #visualize_beats(algorithm_audio_path, algorithm_beats_seconds, sample_rate)

            print(f"Audio File: {file_name}")
            print(f"Algorithm Beats: {algorithm_beats}")
            print(f"Ground Truth Beats: {ground_truth_beats}")            
    
            print('------------------------------')
            num_files += 1

        if num_files > 0:
            print(f'Genre: {genre}')
            print(f'Num Files: {num_files}')
            print('------------------------------')

        for genre, f_measure_sum in genre_f_measure_sums.items():
            genre_average = sum(f_measure_sum) / num_files
            f_measure_data[genre] = genre_average
            total_f_measure_sum += sum(f_measure_sum)


        total_num_files += num_files

    f_measure_data["overall"] = total_f_measure_sum / total_num_files

    save_f_measure_scores(f_measure_data)

# Replace with the paths to your folders containing the beat location files
algorithm_beats_folder = 'Data/genres_original/'
ground_truth_beats_folder = 'gtzan_tempo_beat-main/beats/'

evaluate_beat_tracking_algorithm(algorithm_beats_folder, ground_truth_beats_folder)

