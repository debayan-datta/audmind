import os
import librosa
import soundfile as sf

# Input and output directories
input_dir = "./test_mentalhealth"
output_dir = "./test_mentalhealth_16khz"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all .wav files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load and resample to 16kHz
        audio_signal, _ = librosa.load(input_path, sr=16000)

        # Save the resampled audio
        sf.write(output_path, audio_signal, 16000)

        print(f"Resampled and saved: {output_path}")
