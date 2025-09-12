import os
from typing import List, Literal
import re
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset, get_template
from swift.plugin import InferStats
import pandas as pd
import json
from ast import literal_eval
from collections import defaultdict
import warnings
import random

def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=2048, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    output = ""
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
        output += resp_list[0].choices[0].delta.content
    print()
    print(f'metric: {metric.compute()}')
    return output


def get_message(audiopath, prompt, system_content, transcript, tool_check):
    if tool_check == True:
        conversation = [
                        {'role': 'system', 'content': system_content},
                        {'role': 'user', 'content': [
                            {"type": "audio", "audio_url": audiopath},
                            {"type": "text", "text": f"Transcription of the audio: {transcript}"},
                            {"type": "text", "text": f"Question: {prompt}"},
                        ]}]
    else:
         conversation = [
                        {'role': 'system', 'content': system_content},
                        {'role': 'user', 'content': [
                            {"type": "audio", "audio_url": audiopath},
                            {'type': 'text', 'text': f"Transcription of audio is: {transcript}"},
                            {"type": "text", "text": f"Question: {prompt}"}
                        ]}]
    return conversation

model = 'qwen2_audio'
last_model_checkpoint = "/data/amey_2311cs10/AudioLM_baselines/audio-reasoner/Audio-Reasoner" #Please replace it with the path to checkpoint
# engine = PtEngine(last_model_checkpoint, max_batch_size=64,  model_type = model)
engine = PtEngine(
    model_id_or_path=last_model_checkpoint,
    model_type=model,
    max_batch_size=64,
    from_pretrained=False
)

def audioreasoner_gen(audiopath, prompt, system_content, transcript, tool_check):
    messages=get_message(audiopath, prompt, system_content, transcript, tool_check)
    return infer_stream(engine, InferRequest(messages))


def main():
    df = pd.read_csv("./audmind_test_tool_values.csv")
    train_df = pd.read_csv("./audmind_train_tools_values.csv")

    with open("./matched_tool_lists.json", "r") as f:
        raw_dict = json.load(f)

    processed_dict = {}
    for key_str, value in raw_dict.items():
        key_set = frozenset(literal_eval(key_str))  # frozenset is hashable
        processed_dict[key_set] = value

    system_content_with_tool = """You are an expert audio analysis assistant that can process audio files, read the transcription and understand questions to provide comprehensive answer. 

Your task is to:
1. Listen to the provided audio file and read the transcript
2. Analyze the question being asked

OUTPUT FORMAT:
You must provide your response in the following exact format:
[Your answer to the question based on the audio content and the transcription]

The final answer should be in the format like below the examples:
{context}

GUIDELINES:
- Be thorough and analytical in your responses
- Maintain objectivity
- Provide clear, actionable insights
- Consider both audio content and transcripton"""

    system_content_without_tool = """You are an expert audio analysis assistant that can process audio files, read the transcription and understand questions to provide comprehensive answer. 

Your task is to:
1. Listen to the provided audio file and read the transcript
2. Analyze the question being asked

Your answer to the question should be based on the audio content and the transcription.

GUIDELINES:
- Be thorough and analytical in your responses
- Maintain objectivity
- Provide clear, actionable insights
- Consider both audio content and transcripton"""        

    
    outputs = []

    for idx, row in df.iterrows():
        audiopath = row['file_id']
        prompt = row["Question"]
        transcript = row['text']

        tool_list = row['selected tools_list']
        set_tool_list  = frozenset(literal_eval(tool_list))

        # Efficient lookup
        selected_items = []
        if set_tool_list in processed_dict:
            values = processed_dict[set_tool_list]
            selected_items = random.sample(values, min(10, len(values)))   # sample of 10 rows for context learning
        
        print(f"\n=== Running inference for file_id: {row['file_id']} ===")
        try:
            if selected_items:
                subset_df = train_df[train_df['file_id'].isin(selected_items)].copy()
                context_examples = "\n".join(
                    f"{i+1}. {ans}" for i, ans in enumerate(subset_df['Answer']))

                system_content = system_content_with_tool.format(context=context_examples)
                tool_check = True
                response = audioreasoner_gen(audiopath, prompt, system_content, transcript, tool_check)
            
            else:
                system_content = system_content_without_tool
                tool_check = False
                response = audioreasoner_gen(audiopath, prompt, system_content, transcript, tool_check)

              
        except Exception as e:
            print(f"Error processing {row['file_id']}: {e}")
            response = "ERROR"

        outputs.append(response)

    # Add the responses to the original DataFrame
    df['ModelResponse'] = outputs

    # Save to a new CSV (or overwrite original if you want)
    output_csv_path = "ar_inf_incontextlearning.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nAll outputs saved to {output_csv_path}")


if __name__ == '__main__':
    main()

