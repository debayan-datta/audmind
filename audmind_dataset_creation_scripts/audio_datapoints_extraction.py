import os
import pandas as pd
import subprocess
from glob import glob
from pydub import AudioSegment
# import ffmpeg


# Directories
qar_dir = "Transcriptions_qar"
audio_base_dir = "audiochunks_all"
output_dir = "audio_datasets"

os.makedirs(output_dir, exist_ok=True)

# # Process each QAR CSV
# for csv_path in glob(os.path.join(qar_dir, "*_qar.csv")):
#     audio_id = os.path.basename(csv_path).replace("_qar.csv", "")
#     print(f"Processing: {audio_id}")

#     df = pd.read_csv(csv_path)
#     # df = df.fillna(-1)  # To handle missing chunk, begin, end times

#     # Track datapoint number for each chunk
#     current_chunk = None
#     datapoint_counter = 0

#     for _, row in df.iterrows():
#         chunk = int(row["chunk"]) if row["chunk"] != -1 else None
#         begin = float(row["Beginning timestamp"]) if row["Beginning timestamp"] != -1 else None
#         end = float(row["End timestamp"]) if row["End timestamp"] != -1 else None

#         if chunk is None or begin is None or end is None:
#             continue

#         # Check if this row has any task marked as 1
#         tasks_to_extract = []
#         for task_num in range(1, 5):
#             if int(row.get(f"task{task_num}", 0)) == 1:
#                 tasks_to_extract.append(task_num)

#         if not tasks_to_extract:
#             continue  # skip if no task is marked

#         # If chunk changed, reset datapoint
#         if current_chunk != chunk:
#             current_chunk = chunk
#             datapoint_counter = 1
#         else:
#             datapoint_counter += 1

#         # Calculate actual start and end time in the full audio
#         abs_start = (chunk - 1) * 30 + begin
#         abs_end = (chunk - 1) * 30 + end
#         duration = abs_end - abs_start

#         # Path to source audio
#         wav_filename = f"{audio_id}_part{chunk}.wav"
#         wav_path = os.path.join(audio_base_dir, audio_id, wav_filename)

#         if not os.path.isfile(wav_path):
#             print(f"Missing audio file: {wav_path}")
#             continue

#         # Extract and save for each relevant task
#         for task_no in tasks_to_extract:
#             out_name = f"{audio_id}_chunk{chunk}_data{datapoint_counter}_task{task_no}.wav"
#             out_path = os.path.join(output_dir, out_name)

#             try:
#                 audio = AudioSegment.from_wav(wav_path)
#                 clip = audio[abs_start * 1000 : abs_end * 1000]  # pydub works in milliseconds
#                 clip.export(out_path, format="wav")
#                 # print(f"Saved: {out_path}")
#             except Exception as e:
#                 print(f"Failed to extract {out_path}: {e}")            

from tqdm import tqdm

# Process each QAR CSV
csv_files = glob(os.path.join(qar_dir, "*_qar.csv"))
for csv_path in tqdm(csv_files, desc="CSV Files"):
# for csv_path in glob(os.path.join(qar_dir, "*_qar.csv")):
    audio_id = os.path.basename(csv_path).replace("_qar.csv", "")
    print(f"Processing: {audio_id}")

    df = pd.read_csv(csv_path)
    # df = df.fillna(-1)  # To handle missing chunk, begin, end times

    # Track datapoint number for each chunk
    current_chunk = None
    datapoint_counter = 0

    

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {audio_id}"):
    # for _, row in df.iterrows():
        chunk = int(row["chunk"]) if row["chunk"] != -1 else None
        begin = float(row["Beginning timestamp"]) if row["Beginning timestamp"] != -1 else None
        end = float(row["End timestamp"]) if row["End timestamp"] != -1 else None

        if chunk is None or begin is None or end is None:
            continue

        # Check if this row has any task marked as 1
        tasks_to_extract = []
        for task_num in range(1, 5):
            if int(row.get(f"task{task_num}", 0)) == 1:
                tasks_to_extract.append(task_num)

        if not tasks_to_extract:
            continue  # skip if no task is marked

        # If chunk changed, reset datapoint
        if current_chunk != chunk:
            current_chunk = chunk
            datapoint_counter = 1
        else:
            datapoint_counter += 1

        # Calculate actual start and end time in the full audio
        abs_start = (chunk - 1) * 30 + begin
        abs_end = (chunk - 1) * 30 + end
        duration = abs_end - abs_start

        # Path to source audio
        wav_filename = f"{audio_id}_part{chunk}.wav"
        wav_path = os.path.join(audio_base_dir, audio_id, wav_filename)

        if not os.path.isfile(wav_path):
            print(f"Missing audio file: {wav_path}")
            continue

        # Extract and save for each relevant task
        for task_no in tasks_to_extract:
            out_name = f"{audio_id}_chunk{chunk}_data{datapoint_counter}_task{task_no}.wav"
            out_path = os.path.join(output_dir, out_name)

            try:
                audio = AudioSegment.from_wav(wav_path)
                clip = audio[abs_start * 1000 : abs_end * 1000]  # pydub works in milliseconds
                clip.export(out_path, format="wav")
                # print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed to extract {out_path}: {e}")   

