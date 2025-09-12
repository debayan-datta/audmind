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
import warnings
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
        self.system_prompt = self._create_system_prompt()
        
        # Initialize model
        self._initialize_model()
    
    def _create_system_prompt(self):
        """Create the comprehensive system prompt for audio analysis"""
        return """You are an expert audio analysis assistant that can process audio files, read the transcription, understand questions, and utilize tool analysis values to provide comprehensive answer. The tool information should only be used for understanding the context and helping with the audio and transcription but the tool information shouldnt be included in the output.

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
- Consider both audio content and quantitative measurements from tools"""
    
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
    
    def process_dataframe(self, df, batch_size=8, save_progress=True, output_file="audmind_test__alltools_qwen_results_only_answer.csv"):
        """
        Process entire dataframe with audio analysis
        
        Args:
            df (pd.DataFrame): Input dataframe
            batch_size (int): Batch size for processing
            save_progress (bool): Whether to save progress
            output_file (str): Output file name
            
        Returns:
            pd.DataFrame: Processed dataframe with results
        """
        logger.info(f"Starting to process {len(df)} rows")
        
        # Validate required columns
        required_columns = ['file_id', 'text', 'Question', 'Answer', 'Reasoning', 'all toolvalues', 'selected toolvalues']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create result dataframe
        result_df = df.copy()
        result_df['predicted_answer'] = ""
        result_df['predicted_reasoning'] = ""
        result_df['processing_status'] = ""
        result_df['raw_response'] = ""
        
        # Process in batches
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} (rows {start_idx}-{end_idx})")
            
            # Prepare conversations for batch
            batch_conversations = []
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                
                # Format tool values
                formatted_tool_values = self._format_tool_values(row['all toolvalues']) #change this later
                
                conversation = [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': [
                        {"type": "audio", "audio_url": row['file_id']},
                        {"type": "text", "text": f"Transcription of the audio: {row['text']}"},
                        {"type": "text", "text": f"Question: {row['Question']}"},
                        {"type": "text", "text": f"Tool Analysis Values: {formatted_tool_values}"}
                    ]}
                ]
                batch_conversations.append(conversation)
            
            # Process batch
            results = self.process_batch(batch_conversations, batch_size=len(batch_conversations))
            
            # Update result dataframe
            for i, result in enumerate(results):
                actual_idx = start_idx + i
                result_df.loc[actual_idx, 'predicted_answer'] = result['answer']
                result_df.loc[actual_idx, 'predicted_reasoning'] = result['reasoning']
                result_df.loc[actual_idx, 'processing_status'] = result['status']
                result_df.loc[actual_idx, 'raw_response'] = result.get('raw_response', '')
            
            # Save progress periodically
            # if save_progress and (batch_idx + 1) % 5 == 0:
            #     progress_file = f"progress_{batch_idx + 1}_{output_file}"
            #     result_df.to_csv(progress_file, index=False)
            #     logger.info(f"Progress saved to {progress_file}")
        
        # Save final results
        result_df.to_csv(output_file, index=False)
        logger.info(f"Processing complete. Results saved to {output_file}")
        
        return result_df
    
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
    
    df = pd.read_csv("audmind_test_toolvalues_qwen.csv")
    #df = df.head()
    
    # Initialize processor
    processor = AudioAnalysisBatchProcessor()
    
    file_to_be_saved_as = "audmind_test__alltools_qwen_results_only_answer.csv"
    # Process dataframe
    result_df = processor.process_dataframe(
        df, 
        batch_size=2, 
        save_progress=True,
        output_file="audmind_test__alltools_qwen_results_only_answer.csv"
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
