import pandas as pd
import IPython.display as ipd
import os

import openai


openai_api_key = 'your-api-key'


## Make a single transcript file in a proper dialgoue transcript -------------------------------------------------

def merge_consecutive_speaker_rows(df):
    # Compute end timestamp
    df["End timestamp"] = df["Beginning timestamp"] + df["duration"]
    
    # Merging logic
    merged_data = []
    current_speaker = df.iloc[0]["SpeakerID"]
    current_text = df.iloc[0]["text"]
    current_start = df.iloc[0]["Beginning timestamp"]
    current_end = df.iloc[0]["End timestamp"]
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        if row["SpeakerID"] == current_speaker:
            # Concatenate text and update end timestamp
            current_text += " " + row["text"]
            current_end = row["End timestamp"]
        else:
            # Append current merged entry
            merged_data.append([current_speaker, current_text, current_start, current_end])
            # Start a new merge
            current_speaker = row["SpeakerID"]
            current_text = row["text"]
            current_start = row["Beginning timestamp"]
            current_end = row["End timestamp"]
    
    # Append the last merged entry
    merged_data.append([current_speaker, current_text, current_start, current_end])
    
    # Create merged DataFrame
    merged_df = pd.DataFrame(merged_data, columns=["SpeakerID", "text", "Beginning timestamp", "End timestamp"])
    return merged_df


## Apply column Speaker Doctor/Patient ---------------------------------------------------------

def classify_speakers(merged_df, api_key):
    client = openai.OpenAI(api_key=api_key)  # Initialize OpenAI client
    
    # Create a full conversation context
    conversation = "\n".join(
        [f"{row['SpeakerID']}: {row['text']}" for _, row in merged_df.iterrows()]
    )
    
    results = []
    for _, row in merged_df.iterrows():
        prompt = f"""
        The following is a full conversation between a Doctor and a Patient. Identify the speaker type as either 'Doctor' or 'Patient' for each sentence while considering the entire conversation context.
        
        Conversation:
        {conversation}
        
        Now classify the following sentence:
        
        Speaker: {row['SpeakerID']}
        Text: {row['text']}
        
        Return only 'Doctor' or 'Patient'.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that classifies speakers in medical conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        classification = response.choices[0].message.content.strip()
        results.append(classification)
    
    merged_df["Speaker"] = results  # Add predicted column
    return merged_df

## Generalize for all the csv chunks for that video ---------------------------------------------------------

def process_transcriptions(folder_path, api_key):
    final_df = pd.DataFrame()

    # Get all filenames and sort them numerically
    transcript_parts = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_part')[1].split('.csv')[0]))
    
    for transcript_part in transcript_parts:
        pos = transcript_parts.index(transcript_part) + 1              # to get the chunk number
        transcript_path = os.path.join(folder_path, transcript_part)   # access the csv file
        
        df = pd.read_csv(transcript_path)
        # Merge consecutive speaker rows
        merged_df = merge_consecutive_speaker_rows(df)
        # Classify speakers
        merged_df = classify_speakers(merged_df, api_key)

        # Add meta data
        merged_df["chunk"] = pos
        
        # Append to final DataFrame
        final_df = pd.concat([final_df, merged_df], ignore_index=True)

    return final_df



# Perform above task for all the audio-transcripts from different YT vids ---------------------------------------------------------

def process_all_transcripts(base_folder, output_folder, api_key):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all subdirectories (each is a transcript_id folder)
    for transcript_id in os.listdir(base_folder):
        transcript_path = os.path.join(base_folder, transcript_id)

        if os.path.isdir(transcript_path):
            print(f"Processing {transcript_id}...")

            try:
                final_df = process_transcriptions(transcript_path, api_key)

                # Save the final_df to output folder
                output_file = os.path.join(output_folder, f"{transcript_id}.csv")
                final_df.to_csv(output_file, index=False)
                print(f"Saved: {output_file}")
            except Exception as e:
                print(f"Failed to process {transcript_id}: {e}")


if __name__ == '__main__':
    base_folder = "Transcriptions_all"
    output_folder = "Transcriptions_conversations"
    process_all_transcripts(base_folder, output_folder, openai_api_key)

