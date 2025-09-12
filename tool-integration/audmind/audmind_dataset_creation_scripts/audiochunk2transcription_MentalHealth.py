import os
import shutil
import torch
import torchaudio
import pandas as pd
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
# import ffmpeg
import zipfile

import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import login
login(token="<your_token>")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} with dtype: {torch_dtype}")


model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipe_audio2transcript = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

pipeline_diar = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="<use_auth_token>"
)
pipeline_diar.to(torch.device(device))


def read_rttm(file_path):
    rttm_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                rttm_data.append({
                    'filename': parts[1],
                    'channel': parts[2],
                    'start_time': float(parts[3]),
                    'duration': float(parts[4]),
                    'speaker_id': parts[7],
                })
    return rttm_data


def transcribe_segments(audio_path, rttm_data):
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error reading audio file {audio_path}: {e}")
        return pd.DataFrame()  # Return empty dataframe to skip

    results = []

    for entry in rttm_data:
        start_ms = int(entry["start_time"] * 1000)
        end_ms = start_ms + int(entry["duration"] * 1000)
        segment = audio[start_ms:end_ms]
        segment.export("segmented_audio.wav", format="wav")
        try:
            transcription = pipe_audio2transcript("segmented_audio.wav")["text"]
        except Exception as e:
            print(f"Error transcribing segment of {audio_path}: {e}")
            continue
        results.append({
            "SpeakerID": entry["speaker_id"],
            "text": transcription,
            "Beginning timestamp": entry["start_time"],
            "duration": entry["duration"],
        })
    return pd.DataFrame(results)


def process_audios(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for audio_id in os.listdir(input_dir):
        audio_path = os.path.join(input_dir, audio_id)
        print("working on Audio:", audio_path)

        if os.path.isdir(audio_path):
            transcription_folder = os.path.join(output_dir, audio_id)
            os.makedirs(transcription_folder, exist_ok=True)

            for wav_file in sorted(os.listdir(audio_path)):
                if wav_file.endswith(".wav"):
                    full_wav_path = os.path.join(audio_path, wav_file)
                    rttm_file = os.path.join(transcription_folder, wav_file.replace(".wav", ".rttm"))

                    try:
                        # Run diarization
                        diarization = pipeline_diar(full_wav_path, num_speakers=2)
                        with open(rttm_file, "w") as rttm:
                            diarization.write_rttm(rttm)

                        # Read RTTM and transcribe
                        rttm_data = read_rttm(rttm_file)
                        df = transcribe_segments(full_wav_path, rttm_data)

                        if df.empty:
                            print(f"Skipping {wav_file} due to empty transcription.")
                            os.remove(rttm_file)
                            continue

                        # Save transcription CSV
                        csv_path = os.path.join(transcription_folder, wav_file.replace(".wav", ".csv"))
                        df.to_csv(csv_path, index=False)

                    except Exception as e:
                        print(f"Error processing file {full_wav_path}: {e}")

                    # Clean up RTTM file
                    if os.path.exists(rttm_file):
                        os.remove(rttm_file)
                        
                        
def extract_zip(zip_path):
    # Extract the zip file into the current directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()  # Default is current working directory

    print(f"Contents of {zip_path} extracted in current directory")
    
    
if __name__ == "__main__":
    print("Process Begins #################")

    # Step 1: Extract the ZIP archive
    # zip_file_path = "audioset_2.zip"  # <-- renamed
    # # zip_file_path = '/content/drive/MyDrive/Audios_part4_1.zip'
    # extracted_path = "Audios_extracted"
    # extract_zip(zip_file_path, extracted_path)

    zip_file_path = "audiochunks_all.zip"  # path to your zip file
    extract_zip(zip_file_path)
    extracted_path = "audiochunks_all"
    # Step 2: Run your original processing on the extracted directory
    output_directory = "Transcriptions_all"
    process_audios(extracted_path, output_directory)

    print("Process completed ###############")
    
    
    # Define the path to the 'Transcriptions_subset' directory
    source_dir = 'Transcriptions_all'
    # Define the target zip file path
    target_zip = 'Transcriptions_all.zip'
    # Create the zip file, including only the contents of Transcriptions_subset, not the 'content' parent folder
    shutil.make_archive(target_zip.replace('.zip', ''), 'zip', source_dir)
    
    