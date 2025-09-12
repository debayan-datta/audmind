import os
from typing import List, Literal
import re
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset, get_template
from swift.plugin import InferStats
import pandas as pd

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


def get_message(audiopath, prompt, system_content, tool_analysis, transcript):
    messages = [
        {"role": "system", "content": system_content},
        {'role': 'user', 'content': [
            {'type': 'audio', 'audio': audiopath},
            {'type': 'text', 'text': f"Transcription of audio is: {transcript}"},
            {'type': 'text', 'text': f"Question is: {prompt}"},
            {'type': 'text', 'text': f"The tool analysis is as follows = {tool_analysis}"}
        ]
        }
    ]
    return messages

model = 'qwen2_audio'
last_model_checkpoint = "/data/amey_2311cs10/AudioLM_baselines/audio-reasoner/Audio-Reasoner" #Please replace it with the path to checkpoint
# engine = PtEngine(last_model_checkpoint, max_batch_size=64,  model_type = model)
engine = PtEngine(
    model_id_or_path=last_model_checkpoint,
    model_type=model,
    max_batch_size=64,
    from_pretrained=False
)

def audioreasoner_gen(audiopath, prompt, system_content, tool_analysis, transcript):
    return infer_stream(engine, InferRequest(messages=get_message(audiopath, prompt, system_content, tool_analysis, transcript)))


def main():
    csv_path = "/data/amey_2311cs10/AudioLM_baselines/audio-reasoner/audmind_test_toolvalues_qwen.csv"  # Replace with your actual CSV path
    df = pd.read_csv(csv_path)

    outputs = []

    for idx, row in df.iterrows():
        audiopath = row['file_id']
        prompt = row["Question"]
        tool_analysis = row['selected toolvalues']
        transcript = row['text']
        system_content = """You are an expert audio analysis assistant that can process audio files, read the transcription, understand questions, and utilize tool analysis values to provide comprehensive answer. The tool information should only be used for understanding the context and helping with the audio and transcription but the tool information shouldnt be included in the output.

Your task is to:
1. Listen to the provided audio file and read the transcription
2. Analyze the question being asked
3. Utilize the provided tool analysis values to inform your understanding

TOOL VALUES INFORMATION:
The tool values provided contain various audio analysis metrics that can help inform your understanding:
- Acoustic features (pitch, speech rate, pauses)
- Spectral characteristics (MFCC, LPC coefficients)
- Emotional and sentiment analysis results
- Voice quality measurements
- Depression and mental health indicators

OUTPUT FORMAT:
You must provide your response in the following exact format:
[Your answer to the question based on the audio content, the transcription and tool analysis]

The final answer should be in the fomrat like below the examples:
1. Yes, this wellness dimension exists here.
2. Yes, the patient suffers from stress.
3. This shows mental disorder symptoms related to anxiety.

GUIDELINES:
- The Answer part shouldnot contain info from the Tool Analysis.
- Be thorough and analytical in your responses
- Reference relevant tool values when they support your conclusions
- Maintain objectivity and scientific rigor
- Provide clear, actionable insights
- Consider audio content, the transcription and quantitative measurements from tools"""

        print(f"\n=== Running inference for file_id: {row['file_id']} ===")
        try:
            response = audioreasoner_gen(audiopath, prompt, system_content, tool_analysis, transcript)
        except Exception as e:
            print(f"Error processing {row['file_id']}: {e}")
            response = "ERROR"

        outputs.append(response)

    # Add the responses to the original DataFrame
    df['ModelResponse'] = outputs

    # Save to a new CSV (or overwrite original if you want)
    output_csv_path = "ar_inf_selecttools_audmind_only_answer.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nAll outputs saved to {output_csv_path}")


if __name__ == '__main__':
    main()


# import pandas as pd
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# # Load the model directly from Hugging Face (no local download needed)
# MODEL_ID = "zhifeixie/Audio-Reasoner"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("Loading model...")
# processor = AutoProcessor.from_pretrained(MODEL_ID)
# model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(DEVICE)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     return_timestamps=False,
#     device=0 if DEVICE == "cuda" else -1,
# )

# def audioreasoner_gen(audio_path, question, answer, reasoning):
#     # Process audio and get model transcript
#     transcript = pipe(audio_path)["text"]
    
#     # Create system prompt for reasoning
#     system_prompt = f"<Answer>{answer}</Answer><Reasoning>{reasoning}</Reasoning>"
    
#     # Final output (can be extended to prompt a second model for reasoning)
#     return f"{transcript}\n\nSystem Expected:\n{system_prompt}"

# def main():
#     csv_path = "your_csv_file.csv"  # Replace with your actual CSV path
#     df = pd.read_csv(csv_path, sep="\t")

#     outputs = []

#     for idx, row in df.iterrows():
#         audio_path = f"/data/amey_2311cs10/AudioLM_baselines/GAMA/test_mentalhealth_16kHz/{row['file_id']}.wav"
#         question = row["Question"]
#         answer = row["Answer"]
#         reasoning = row["Reasoning"]

#         print(f"\n=== Running inference for file_id: {row['file_id']} ===")
#         try:
#             response = audioreasoner_gen(audio_path, question, answer, reasoning)
#         except Exception as e:
#             print(f"Error processing {row['file_id']}: {e}")
#             response = "ERROR"

#         outputs.append(response)

#     df["ModelResponse"] = outputs
#     df.to_csv("output_audio_reasoner.csv", index=False)
#     print("\nAll outputs saved to output_audio_reasoner.csv")

# if __name__ == "__main__":
#     main()