import os
import pandas as pd
import soundfile as sf

# Define base directories
csv_dir = "Transcriptions_qar"
audio_base_dir = "audiochunks_all"
# output_root = "final_audio_dataset"
output_dir = "final_audio_dataset"

# # Create output root directory if it doesn't exist
# if not os.path.exists(output_root):
#     os.makedirs(output_root)

# Create output dir directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over all CSV files in Transcriptions_qar
for csv_filename in os.listdir(csv_dir):
    if not csv_filename.endswith("_qar.csv"):
        continue

    csv_path = os.path.join(csv_dir, csv_filename)

    # Extract audio_id from CSV filename
    audio_id = csv_filename.replace("_qar.csv", "")

    # Define corresponding audio chunk directory
    base_dir = os.path.join(audio_base_dir, audio_id)

    # # Define output directory for this audio_id
    # output_dir = os.path.join(output_root, audio_id)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f'Currently accessing {audio_id}__________________________________________')
    # Group by 'chunk'
    grouped = df.groupby("chunk")

    # Process each chunk group
    for chunk_num, group in grouped:
        print(f"[{audio_id}] chunk no. = {chunk_num}")
        datapoint = 0
        audio_chunk_file = f"{audio_id}_part{chunk_num}.wav"
        audio_chunk_path = os.path.join(base_dir, audio_chunk_file)

        if not os.path.exists(audio_chunk_path):
            print(f"Warning: {audio_chunk_path} not found.")
            continue

        # Load audio
        data, sr = sf.read(audio_chunk_path)

        for _, row in group.iterrows():
            tasks_active = [task for task in range(1, 5) if row.get(f"task{task}", 0) == 1]
            if not tasks_active:
                continue

            start_sec = float(row["Beginning timestamp"])
            end_sec = float(row["End timestamp"])
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)

            cropped_audio = data[start_frame:end_frame]
            datapoint += 1

            for taskno in tasks_active:
                filename = f"{audio_id}_chunk{chunk_num}_data{datapoint}_task{taskno}.wav"
                output_path = os.path.join(output_dir, filename)
                sf.write(output_path, cropped_audio, sr)
                print(f"--------file saved as {filename}--------")

print("All audio chunks processed. Output saved in:", output_dir)
