import os
import madmom
import librosa
from madmom.features.beats import BeatTrackingProcessor, DBNBeatTrackingProcessor
import numpy as np
import matplotlib.pyplot as plt

def visualize_beats(audio_path, algorithm_beats, sample_rate):
  """
  Visualizes the predicted beats on a time series plot.

  Args:
      audio_path: Path to the audio file. (for informative plot title)
      algorithm_beats: List of beat locations predicted by the algorithm.
      sample_rate: Sample rate of the audio file.
  """
  # Calculate audio duration
  audio_duration = len(algorithm_beats) / sample_rate

  # Create the plot
  plt.figure(figsize=(10, 5))
  plt.plot(np.linspace(0, audio_duration, len(algorithm_beats)), np.zeros(len(algorithm_beats)), color='gray')
  plt.scatter(algorithm_beats / sample_rate, np.ones(len(algorithm_beats)), marker='o', color='red', label="Predicted Beats")

  # Set labels and title
  plt.xlabel("Time (s)")
  plt.ylabel("Beat Probability")
  plt.title(f"Algorithm Beats for: {audio_path}")
  plt.legend()
  plt.grid(True)

  # Display the plot
  plt.show()




def calculate_beats(audio_file_path):
    # Load audio data using librosa
    print(audio_file_path)
    audio, sample_rate = librosa.load(audio_file_path, sr=None)  # Assuming you have librosa installed

    # Use DBNBeatTrackingProcessor (example)
    proc = DBNBeatTrackingProcessor(observation_lambda=0.1, correct=True, fps=100)
    act = proc(audio)  # Pass the loaded audio data (audio)
    beats = proc(act)
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

    for genre in genres:
        genre_path = os.path.join(algorithm_beats_folder, genre)
        if not os.path.isdir(genre_path):
            continue

        genre_files = os.listdir(genre_path)

        num_files = 0

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

# Replace with the paths to your folders containing the beat location files
algorithm_beats_folder = 'Data/genres_original/'
ground_truth_beats_folder = 'gtzan_tempo_beat-main/beats/'

evaluate_beat_tracking_algorithm(algorithm_beats_folder, ground_truth_beats_folder)

