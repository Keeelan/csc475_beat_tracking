import madmom
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from joblib import Parallel, delayed
from sklearn.metrics import precision_recall_fscore_support
import csv
import librosa

process_count = 0


def read_ground_truth_tempo(file_path):
    with open(file_path, 'r') as file:
        # Assuming one tempo value per file
        return float(file.read().strip())


def read_genre(file_path):
    with open(file_path, 'r') as file:
        # Assuming one genre per file
        return file.read().strip()

def calculate_accuracy(predicted_tempo, ground_truth_tempo):
    # Define tolerance levels
    tolerance_accuracy1 = 0.04  # 4% tolerance
    tolerance_accuracy2 = [0.5, 1/2, 1, 2, 3, 1/3]  # Octave errors

    # Check Accuracy1
    if abs(predicted_tempo - ground_truth_tempo) <= tolerance_accuracy1 * ground_truth_tempo:
        accuracy1 = 1
    else:
        accuracy1 = 0

    # Check Accuracy2
    accuracy2 = 0
    for factor in tolerance_accuracy2:
        if abs(predicted_tempo - (factor*ground_truth_tempo)) <= tolerance_accuracy1*ground_truth_tempo:
            accuracy2 = 1
            break

    return accuracy1, accuracy2

def estimate_tempo(audio_file, ground_truth_file, genre_file):

    signal, sample_rate = librosa.load(audio_file)

    fps = 100

    hop_length = int(librosa.time_to_samples(1./fps, sr=sample_rate))

    n_fft = 2048
    n_mels = 80
    fmin = 27.5
    fmax = 17000.
    lag = 2
    max_size = 3


   
    S = librosa.feature.melspectrogram(y = signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)

    spectral_flux = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=sample_rate,
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)

    predicted_tempo = librosa.beat.tempo(onset_envelope=spectral_flux, sr=sample_rate,
                               hop_length=hop_length)[0]

    ground_truth_tempo = read_ground_truth_tempo(ground_truth_file)

    # Read genre from file
    genre = read_genre(genre_file)

    print(f"Estimated tempo for {audio_file}: {predicted_tempo:.2f} BPM - Genre: {genre}")

    # Calculate Accuracy1 and Accuracy2
    accuracy1, accuracy2 = calculate_accuracy(predicted_tempo, ground_truth_tempo)

    print(f"Accuracy1: {accuracy1}")
    print(f"Accuracy2: {accuracy2}")

    return genre, accuracy1, accuracy2, 1  # Return 1 to indicate one file processed

    
if __name__ == '__main__':
    audio_folder_path = 'giantsteps-tempo-dataset/audio/'

    bpm_folder_path = 'giantsteps-tempo-dataset/annotations/tempo/'

    genre_folder_path = 'giantsteps-tempo-dataset/annotations/genre/'

    output_csv_path = 'accuracy.csv'

    # List all .wav files in the audio folder
    audio_files = [file_name for file_name in os.listdir(audio_folder_path) if file_name.endswith('.wav')]

    # Use Joblib with multiprocessing to parallelize tempo estimation
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(estimate_tempo)(
            os.path.join(audio_folder_path, audio_file),
            os.path.join(bpm_folder_path, audio_file.replace('.wav', '.bpm')),
            os.path.join(genre_folder_path, audio_file.replace('.wav', '.genre'))
        )
        for audio_file in audio_files
    )

    # Dictionary to store accuracy results by genre
    genre_results = {}
    for genre, accuracy1, accuracy2, num_files in results:
        if genre not in genre_results:
            genre_results[genre] = {'Accuracy1': 0, 'Accuracy2': 0, 'NumFiles': 0}
        genre_results[genre]['Accuracy1'] += accuracy1
        genre_results[genre]['Accuracy2'] += accuracy2
        genre_results[genre]['NumFiles'] += num_files

    # Calculate average values per genre
    avg_genre_results = []
    for genre, values in genre_results.items():
        avg_accuracy1 = values['Accuracy1'] / values['NumFiles']
        avg_accuracy2 = values['Accuracy2'] / values['NumFiles']
        avg_genre_results.append({'Genre': genre, 'AvgAccuracy1': avg_accuracy1, 'AvgAccuracy2': avg_accuracy2, 'NumFiles': values['NumFiles']})

    # Write average results to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Genre', 'AvgAccuracy1', 'AvgAccuracy2', 'NumFiles']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in avg_genre_results:
            writer.writerow(result)

    print(f"Results saved to {output_csv_path}")
