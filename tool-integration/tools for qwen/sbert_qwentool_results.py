import nltk
nltk.download('punkt')
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # or 'all-mpnet-base-v2' for better quality

# Fallback if nltk not available
def safe_sent_tokenize(text):
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except:
        # Basic sentence split fallback
        return [s.strip() for s in text.split('.') if s.strip()]

# SBERT similarity function with fallback tokenization
def sbert_similarity(text1, text2):
    sents1 = safe_sent_tokenize(text1)
    sents2 = safe_sent_tokenize(text2)

    emb1 = model.encode(sents1, convert_to_tensor=True)
    emb2 = model.encode(sents2, convert_to_tensor=True)

    sim_matrix = util.pytorch_cos_sim(emb1, emb2)
    ref_to_pred = sim_matrix.max(dim=1).values.mean().item()
    pred_to_ref = sim_matrix.max(dim=0).values.mean().item()

    return (ref_to_pred + pred_to_ref) / 2


#-----------------------------------------------------------------------------------------------------------
res = pd.read_csv("audmind_test__alltools_qwen_results_only_answer.csv")

similaritiesA = []
for idx, row in res.iterrows():
    try:
        score = sbert_similarity(row['Answer'], row['predicted_answer'])
    except Exception as e:
        print(f"Error at row {idx}: {e}")
        score = None
    similaritiesA.append(score)

# Compute and print mean similarity (excluding None)
valid_scoresA = [s for s in similaritiesA if s is not None]
mean_scoreA = sum(valid_scoresA) / len(valid_scoresA)


res['ans_reason'] = res['Answer'].str.cat(res['Reasoning'], sep=' ')
similaritiesAR = []
for idx, row in res.iterrows():
    try:
        score = sbert_similarity(row['ans_reason'], row['predicted_answer'])
    except Exception as e:
        print(f"Error at row {idx}: {e}")
        score = None
    similaritiesAR.append(score)

# Compute and print mean similarity (excluding None)
valid_scoresAR = [s for s in similaritiesAR if s is not None]
mean_scoreAR = sum(valid_scoresAR) / len(valid_scoresAR)

print(f"Mean SBERT Semantic Similarity for ANSWER: {mean_scoreA:.4f}")
print(f"Mean SBERT Semantic Similarity for ANSWER + REASONING: {mean_scoreAR:.4f}")