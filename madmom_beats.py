import os
import madmom
import librosa
import soundfile as sf
from madmom.features.beats import BeatTrackingProcessor, RNNBeatProcessor
import numpy as np
import matplotlib.pyplot as plt
import mir_eval
import csv

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
        
    proc = BeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(audio)  # Pass the loaded audio data (audio)
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

            doubled_beats = []
            for beat in algorithm_beats:
                doubled_beats.append(beat * 2)
            
            algorithm_beats_array = np.array(doubled_beats)
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

