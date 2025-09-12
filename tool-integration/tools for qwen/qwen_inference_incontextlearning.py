import pandas as pd
import numpy as np
import json
import ast
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import torch
from tqdm import tqdm
import logging
import os
from ast import literal_eval
from collections import defaultdict
import warnings
import random
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioAnalysisBatchProcessor:
    """
    Comprehensive audio analysis system using Qwen2-Audio for batch processing
    of audio files with questions and tool values to generate structured answers.
    """
    
    def __init__(self, model_name="Qwen/Qwen2-Audio-7B-Instruct", device="auto"):
        """
        Initialize the audio analysis processor
        
        Args:
            model_name (str): The Qwen2-Audio model to use
            device (str): Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.device = device
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.system_prompt_without_tool = self._create_system_prompt_without_tool()
        self.system_prompt_with_tool = self._create_system_prompt_with_tool()
        
        # Initialize model
        self._initialize_model()
    
    def _create_system_prompt_with_tool(self):
        """Create the comprehensive system prompt for audio analysis"""
        return """You are an expert audio analysis assistant that can process audio files, read the transcription and understand questions to provide comprehensive answer. 

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
    
    def _create_system_prompt_without_tool(self):
        """Create the comprehensive system prompt for audio analysis"""
        return """You are an expert audio analysis assistant that can process audio files, read the transcription and understand questions to provide comprehensive answer. 

Your task is to:
1. Listen to the provided audio file and read the transcript
2. Analyze the question being asked

Your answer to the question should be based on the audio content and the transcription.

GUIDELINES:
- Be thorough and analytical in your responses
- Maintain objectivity
- Provide clear, actionable insights
- Consider both audio content and transcripton"""

    
    def _initialize_model(self):
        """Initialize the Qwen2-Audio model and processor with proper configuration"""
        try:
            logger.info(f"Initializing Qwen2-Audio model: {self.model_name}")
            
            # Load processor with explicit sampling rate configuration
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Configure the feature extractor with explicit sampling rate
            if hasattr(self.processor, 'feature_extractor'):
                self.processor.feature_extractor.sampling_rate = 16000
                logger.info(f"Feature extractor sampling rate set to: {self.processor.feature_extractor.sampling_rate}")
            
            # Load model
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _load_audio_file(self, file_path):
        """
        Load audio file and prepare it for processing
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            numpy.ndarray: Audio data array
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            # Load audio using librosa with explicit sampling rate (16kHz)
            audio, sr = librosa.load(file_path, sr=16000)
            
            logger.debug(f"Loaded audio file {file_path} with shape {audio.shape} at {sr}Hz")
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            return None
    
    def _format_tool_values(self, tool_values):
        """
        Format tool values dictionary into a readable string
        
        Args:
            tool_values (dict or str): Tool values dictionary or string
            
        Returns:
            str: Formatted tool values string
        """
        try:
            if isinstance(tool_values, str):
                # Try to parse as dictionary if it's a string
                try:
                    tool_values = ast.literal_eval(tool_values)
                except:
                    # If parsing fails, return as is
                    return tool_values
            
            if isinstance(tool_values, dict):
                formatted_values = []
                for key, value in tool_values.items():
                    if isinstance(value, float):
                        formatted_values.append(f"{key}: {value:.4f}")
                    else:
                        formatted_values.append(f"{key}: {value}")
                return "; ".join(formatted_values)
            else:
                return str(tool_values)
                
        except Exception as e:
            logger.error(f"Error formatting tool values: {str(e)}")
            return str(tool_values)
    
    def process_single_audio(self, file_path, question, tool_values, max_new_tokens=2048):
        """
        Process a single audio file with question and tool values
        
        Args:
            file_path (str): Path to the audio file
            question (str): Question to be answered
            tool_values (dict or str): Tool analysis values
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            dict: Processing result with answer and reasoning
        """
        try:
            # Load audio
            audio = self._load_audio_file(file_path)
            if audio is None:
                return {
                    "answer": "Unable to process audio file",
                    "reasoning": "Audio file could not be loaded or found",
                    "status": "error"
                }

            print("--------Audio loaded------------")
            # Format tool values
            formatted_tool_values = self._format_tool_values(tool_values)
            
            # Create conversation for audio analysis
            conversation = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': [
                    {"type": "audio", "audio_url": file_path},
                    {"type": "text", "text": f"Transcription of audio: {text}"},
                    {"type": "text", "text": f"Question: {question}"},
                    {"type": "text", "text": f"Tool Analysis Values: {formatted_tool_values}"}
                ]}
            ]
            
            # Process the conversation
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Prepare audio input
            audios = [audio]
            
            # Process inputs
            inputs = self.processor(
                text=text, 
                audios=audios, 
                return_tensors="pt", 
                padding=True,
                sampling_rate=16000
            )
            
            # Move to appropriate device
            if torch.cuda.is_available() and self.device != 'cpu':
                inputs.input_ids = inputs.input_ids.to("cuda")
                if hasattr(inputs, 'attention_mask'):
                    inputs.attention_mask = inputs.attention_mask.to("cuda")
            
            # Generate response
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            print("--------raw response generated------------", response)
            
            # Parse the structured response
            parsed_response = self._parse_structured_response(response)
            print("--------parsed response------------", parsed_response)
            return {
                "answer": parsed_response["answer"],
                "reasoning": parsed_response["reasoning"],
                "status": "success",
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {str(e)}")
            return {
                "answer": "Error processing audio",
                "reasoning": f"Processing error: {str(e)}",
                "status": "error"
            }
    
    def _parse_structured_response(self, response):
        """
        Parse the structured response to extract answer and reasoning
        
        Args:
            response (str): Raw response from the model
            
        Returns:
            dict: Parsed answer and reasoning
        """
        try:
            # Extract answer
            answer_start = response.find("<Answer>")
            answer_end = response.find("</Answer>")
            
            if answer_start != -1 and answer_end != -1:
                answer = response[answer_start + 8:answer_end].strip()
            else:
                # Fallback: use first part of response
                answer = response.split("\n")[0].strip()
            
            # Extract reasoning
            reasoning_start = response.find("<Reasoning>")
            reasoning_end = response.find("</Reasoning>")
            
            if reasoning_start != -1 and reasoning_end != -1:
                reasoning = response[reasoning_start + 11:reasoning_end].strip()
            else:
                # Fallback: use remaining response
                reasoning = response.replace(answer, "").strip()
            
            return {
                "answer": answer if answer else "No answer provided",
                "reasoning": reasoning if reasoning else "No reasoning provided"
            }
            
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            return {
                "answer": response[:200] + "..." if len(response) > 200 else response,
                "reasoning": "Could not parse structured response"
            }
    
    def process_batch(self, conversations, batch_size=8, max_new_tokens=2048):
        """
        Process multiple conversations in batch mode
        
        Args:
            conversations (list): List of conversation dictionaries
            batch_size (int): Number of conversations to process in each batch
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            list: List of processing results
        """
        try:
            results = []
            
            # Process in batches
            for i in tqdm(range(0, len(conversations), batch_size), desc="Processing batches"):
                batch_conversations = conversations[i:i+batch_size]
                
                # Prepare batch inputs
                batch_texts = []
                batch_audios = []
                
                for conv in batch_conversations:
                    # Apply chat template
                    text = self.processor.apply_chat_template(
                        conv, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                    batch_texts.append(text)
                    
                    # Extract audio from conversation
                    for message in conv:
                        if isinstance(message.get("content"), list):
                            for content in message["content"]:
                                if content.get("type") == "audio":
                                    audio = self._load_audio_file(content["audio_url"])
                                    if audio is not None:
                                        batch_audios.append(audio)
                
                # Process batch
                inputs = self.processor(
                    text=batch_texts, 
                    audios=batch_audios, 
                    return_tensors="pt", 
                    padding=True,
                    sampling_rate=16000
                )
                
                # Move to device
                if torch.cuda.is_available() and self.device != 'cpu':
                    inputs.input_ids = inputs.input_ids.to("cuda")
                    if hasattr(inputs, 'attention_mask'):
                        inputs.attention_mask = inputs.attention_mask.to("cuda")
                
                # Generate responses
                with torch.no_grad():
                    generate_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode responses
                generate_ids = generate_ids[:, inputs.input_ids.size(1):]
                responses = self.processor.batch_decode(
                    generate_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                # Parse responses
                for response in responses:
                    parsed = self._parse_structured_response(response)
                    results.append({
                        "answer": parsed["answer"],
                        "reasoning": parsed["reasoning"],
                        "status": "success",
                        "raw_response": response
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return [{"answer": "Batch processing error", "reasoning": str(e), "status": "error"}] * len(conversations)

    
    def process_dataframe(self, test, train, processed_dict, batch_size=8, save_progress=True, output_file="audmind_test_qwen_inf_icl.csv"):
        logger.info(f"Starting to process {len(test)} rows")
        # Validate required columns
        required_columns = ['file_id', 'text', 'Question', 'selected tools_list']
        missing_columns = [col for col in required_columns if col not in test.columns]       
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create result dataframe
        result_test = test.copy()
        result_test['predicted_answer'] = ""
        result_test['predicted_reasoning'] = ""
        result_test['processing_status'] = ""
        result_test['raw_response'] = ""
        
        # Process in batches
        total_batches = (len(test) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(test))
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})")
            
            # Prepare conversations for batch
            batch_conversations = []
            
            for idx in range(start_idx, end_idx):
                row = test.iloc[idx]
 
                tool_list = row['selected tools_list']
                set_tool_list  = frozenset(literal_eval(tool_list))
                
                # Efficient lookup
                selected_items = []
                if set_tool_list in processed_dict:
                    values = processed_dict[set_tool_list]
                    selected_items = random.sample(values, min(10, len(values)))   # sample of 10 rows for context learning


                if selected_items:
                    subset_df = train[train['file_id'].isin(selected_items)].copy()
                    context_examples = "\n".join(
                        f"{i+1}. {ans}" for i, ans in enumerate(subset_df['Answer']))
                    
                    conversation = [
                        {'role': 'system', 'content': self.system_prompt_with_tool.format(context=context_examples)},
                        {'role': 'user', 'content': [
                            {"type": "audio", "audio_url": row['file_id']},
                            {"type": "text", "text": f"Transcription of the audio: {row['text']}"},
                            {"type": "text", "text": f"Question: {row['Question']}"},
                        ]}
                    ]

                else:
                    conversation = [
                    {'role': 'system', 'content': self.system_prompt_without_tool},
                    {'role': 'user', 'content': [
                        {"type": "audio", "audio_url": row['file_id']},
                        {"type": "text", "text": f"Question: {row['Question']}"}
                    ]}
                ]

                    
                    
                batch_conversations.append(conversation)
            
            # Process batch
            results = self.process_batch(batch_conversations, batch_size=len(batch_conversations))
            
            # Update result dataframe
            for i, result in enumerate(results):
                actual_idx = start_idx + i
                result_test.loc[actual_idx, 'predicted_answer'] = result['answer']
                result_test.loc[actual_idx, 'predicted_reasoning'] = result['reasoning']
                result_test.loc[actual_idx, 'processing_status'] = result['status']
                result_test.loc[actual_idx, 'raw_response'] = result.get('raw_response', '')
            
        result_test.to_csv(output_file, index=False)
        logger.info(f"Processing complete. Results saved to {output_file}")
        
        return result_test

        
    def analyze_results(self, df):
        """
        Analyze the processing results
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            dict: Analysis results
        """
        if 'processing_status' not in df.columns:
            return {"error": "No processing results found"}
        
        # Calculate success rate
        success_rate = (df['processing_status'] == 'success').mean() * 100
        
        # Calculate average response lengths
        avg_answer_length = df['predicted_answer'].str.len().mean()
        avg_reasoning_length = df['predicted_reasoning'].str.len().mean()
        
        analysis = {
            'total_samples': len(df),
            'success_rate': f"{success_rate:.1f}%",
            'average_answer_length': f"{avg_answer_length:.1f} characters",
            'average_reasoning_length': f"{avg_reasoning_length:.1f} characters",
            'error_count': len(df[df['processing_status'] == 'error'])
        }
        
        return analysis

# Example usage and demo
def main():
    """Main function to demonstrate usage"""

    test_df = pd.read_csv("audmind_test_toolvalues_qwen.csv")
    train_df = pd.read_csv("./tools_selection/audmind_train_tools_values.csv")
    # test_df = test_df.head()
    
    with open("./tools_selection/matched_tool_lists.json", "r") as f:
        raw_dict = json.load(f)

    processed_dict = {}
    for key_str, value in raw_dict.items():
        key_set = frozenset(literal_eval(key_str))  # frozenset is hashable
        processed_dict[key_set] = value
    
    # Initialize processor
    processor = AudioAnalysisBatchProcessor()
    
    file_to_be_saved_as = "audmind_test_qwen_inf_icl.csv" # icl = in context learning
    # Process dataframe
    result_df = processor.process_dataframe(
        test_df,
        train_df,
        processed_dict,
        batch_size=2, 
        save_progress=True,
        output_file="audmind_test_qwen_inf_icl.csv"
    )
    
    # Analyze results
    analysis = processor.analyze_results(result_df)
    
    print("Analysis Results:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    return result_df

if __name__ == "__main__":
    # Run the main function
    result_df = main()
    print(f"\nProcessed {len(result_df)} rows successfully!")
