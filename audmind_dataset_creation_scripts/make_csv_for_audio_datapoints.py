import os
import pandas as pd

# Define base directories
csv_dir = "Transcriptions_qar"
audio_base_dir = "audiochunks_all"

# Define column names for the empty df
df_structured = pd.DataFrame(columns=['file_id', 'Question', 'Answer', 'Reasoning'])



# Extract parts of the QA text
def parse_qa_text(qa_text):
    parts = {}
    for part in ['Question', 'Answer', 'Reasoning']:
        start = qa_text.find(f"{part}:") + len(f"{part}:")
        end = qa_text.find('\n', start)
        if end == -1:
            end = len(qa_text)
        parts[part] = qa_text[start:end].strip()
    return parts
    


# Loop over all CSV files in Transcriptions_qar
for csv_filename in os.listdir(csv_dir):
    if not csv_filename.endswith("_qar.csv"):
        continue
    csv_path = os.path.join(csv_dir, csv_filename)

    # Extract audio_id from CSV filename
    audio_id = csv_filename.replace("_qar.csv", "")
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f'Currently accessing {audio_id} ===========')
    
    # Group by 'chunk'
    grouped = df.groupby("chunk")

    # Process each chunk group
    for chunk_num, group in grouped:
        # print(f"[{audio_id}] chunk no. = {chunk_num}")
        datapoint = 0
        
        for row_idx, row in group.iterrows():
            tasks_active = [task for task in range(1, 5) if row.get(f"task{task}", 0) == 1]
            if not tasks_active:
                continue
            
            datapoint += 1

            for taskno in tasks_active:
                file_id = f"{audio_id}_chunk{chunk_num}_data{datapoint}_task{taskno}"
                
                qa_column = f"QA for task{taskno}"
                qa_text = row.get(qa_column, "")

                if pd.notna(qa_text) and all(x + ":" in qa_text for x in ["Question", "Answer", "Reasoning"]):
                    qa_parts = parse_qa_text(qa_text)

                    df_structured.loc[len(df_structured)] = {
                        "file_id": file_id,
                        "Question": qa_parts["Question"],
                        "Answer": qa_parts["Answer"],
                        "Reasoning": qa_parts["Reasoning"]
                    }
                    
            print(f"---row saved as {file_id}---")


output_path = "structured_output.csv"  # You can change this to your desired filename or path
df_structured.to_csv(output_path, index=False)
print(f"Structured data saved to {output_path}")
