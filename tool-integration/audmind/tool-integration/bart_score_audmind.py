import pandas as pd
import numpy as np
import torch

def calculate_bartscore_similarity(df, column1, column2, use_parabank=True):
    """
    Calculate BARTScore similarity between two DataFrame columns
    
    Parameters:
    - df: pandas DataFrame
    - column1: name of first column
    - column2: name of second column
    - use_parabank: whether to use the ParaBank trained model
    
    Returns:
    - DataFrame with BARTScore column added
    - Mean similarity score
    """
    
    try:
        import sys
        sys.path.append('/path/to/BARTScore')  # Update this path
        from bart_score import BARTScorer
        
        # Device selection
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize scorer
        bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        
        # Load ParaBank model if requested
        if use_parabank:
            try:
                bart_scorer.load(path='bart.pth')
                print("Using ParaBank trained model")
            except:
                print("ParaBank model not found, using default model")
        
        # Convert columns to lists
        text1_list = df[column1].astype(str).tolist()
        text2_list = df[column2].astype(str).tolist()
        
        print(f"Calculating BARTScore for {len(text1_list)} text pairs...")
        
        # Calculate scores in batches
        scores = []
        batch_size = 4
        
        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i+batch_size]
            batch_text2 = text2_list[i:i+batch_size]
            
            try:
                batch_scores = bart_scorer.score(batch_text1, batch_text2, batch_size=batch_size)
                scores.extend(batch_scores)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                scores.extend([0.0] * len(batch_text1))
        
        # Add scores to DataFrame
        result_df = df.copy()
        result_df['bartscore'] = scores
        
        # Calculate mean similarity
        mean_score = np.mean(scores)
        
        return result_df, mean_score
        
    except ImportError:
        print("BARTScore not available. Please install following the setup instructions.")
        return df, 0.0
    except Exception as e:
        print(f"Error in BARTScore calculation: {e}")
        return df, 0.0

# Usage example
#-----------main----------------------
# dfno = pd.read_csv("data/amey_2311cs10/AudioLM_baselines/audio-reasoner/ar_inf_notools_audmind_only_answer.csv")
# dfselect = pd.read_csv("data/amey_2311cs10/AudioLM_baselines/audio-reasoner/ar_inf_selecttools_audmind_only_answer.csv")
# dfall = pd.read_csv("data/amey_2311cs10/AudioLM_baselines/audio-reasoner/ar_inf_alltools_audmind_only_answer.csv")

# dfno, mean_score_no = calculate_bartscore_similarity(dfno, 'Answer', 'ModelResponse')
# dfselect, mean_score_select = calculate_bartscore_similarity(dfselect, 'Answer', 'ModelResponse')
# dfall, mean_score_all = calculate_bartscore_similarity(dfall, 'Answer', 'ModelResponse')

# print("---------- ONLY ANSWER ---------------")
# print(f"Overall similarity score for NO Tools: {mean_score_no:.4f}")
# print(f"Overall similarity score for Selected Tools: {mean_score_select:.4f}")
# print(f"Overall similarity score for All Tools: {mean_score_all:.4f}")

# dfno['ans_reason'] = dfno['Answer'].str.cat(dfno['Reasoning'], sep=' ')
# dfselect['ans_reason'] = dfselect['Answer'].str.cat(dfselect['Reasoning'], sep=' ')
# dfall['ans_reason'] = dfall['Answer'].str.cat(dfall['Reasoning'], sep=' ')

# dfno, mean_score_no = calculate_bartscore_similarity(dfno, 'ans_reason', 'ModelResponse')
# dfselect, mean_score_select = calculate_bartscore_similarity(dfselect, 'ans_reason', 'ModelResponse')
# dfall, mean_score_all = calculate_bartscore_similarity(dfall, 'ans_reason', 'ModelResponse')

# print("---------- ANSWER + REASON ---------------")
# print(f"Overall similarity score for NO Tools: {mean_score_no:.4f}")
# print(f"Overall similarity score for Selected Tools: {mean_score_select:.4f}")
# print(f"Overall similarity score for All Tools: {mean_score_all:.4f}")

df_icl = pd.read_csv("./Qwen2Audio/audmind_test_qwen_inf_icl.csv")

df_icl, mean_score_icl = calculate_bartscore_similarity(df_icl, 'Answer', 'predicted_answer')
print("---------- ONLY ANSWER ---------------")
print(f"Overall similarity score: {mean_score_icl:.4f}")


df_icl['ans_reason'] = df_icl['Answer'].str.cat(df_icl['Reasoning'], sep=' ')
df_icl, mean_score_ar_icl = calculate_bartscore_similarity(df_icl, 'ans_reason', 'predicted_answer')
print("---------- ANSWER + REASON ---------------")
print(f"Overall similarity score:: {mean_score_ar_icl:.4f}")

