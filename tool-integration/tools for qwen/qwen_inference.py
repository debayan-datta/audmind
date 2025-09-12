import json
from tqdm import tqdm
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import pandas as pd

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

# Load your JSON
with open("qwen2audio_inference.json", "r") as f:
    data = json.load(f)

# Store rows for output
output_rows = []

for item in tqdm(data):
    conversation = item["conversation"]

    # 1. Process text
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # 2. Load audio
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    # audio_data = urlopen(ele["audio_url"]).read()
                    # y, _ = librosa.load(BytesIO(audio_data), sr=processor.feature_extractor.sampling_rate)
                    y, _ = librosa.load(ele["audio_url"], sr=processor.feature_extractor.sampling_rate)
                    
                    audios.append(y)

    # 3. Prepare inputs
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 4. Generate
    generate_ids = model.generate(**inputs, max_length=1024)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]  # skip prompt tokens
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # 5. Extract original metadata (from audio filename and question)
    user_content = next(m for m in conversation if m["role"] == "user")["content"]
    file_id = next(ele["audio_url"] for ele in user_content if ele["type"] == "audio").split("/")[-1].replace(".wav", "")
    question = next(ele["text"] for ele in user_content if ele["type"] == "text")

    # 6. Append result
    output_rows.append({
        "file_id": file_id,
        "Question": question,
        "Generated": response
    })
    
    print(response)

# Convert to DataFrame
df_output = pd.DataFrame(output_rows)
print(df_output.head())

# # Optionally, merge with original CSV if you want to include gold Answer/Reasoning:
# df_original = pd.read_csv("your_file.csv")  # or ',' depending on original
# df_merged = df_original.merge(df_output, on=["file_id", "Question"])

# # Save to CSV
# df_merged.to_csv("qwen_inf_output.csv", index=False)
df_output.to_csv("qwen_inf_output.csv", index=False)
