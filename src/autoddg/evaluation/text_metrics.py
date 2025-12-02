from rouge_score import rouge_scorer
from bert_score import score as bertscore
from nltk.translate.meteor_score import meteor_score


from sklearn.feature_extraction.text import CountVectorizer



#---- Reference-based text metrics ---
def compute_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r = scorer.score(ref, pred)
    return {
        "rouge1": r["rouge1"].fmeasure,
        "rouge2": r["rouge2"].fmeasure,
        "rougeL": r["rougeL"].fmeasure,
    }

def compute_meteor(pred, ref):
    return meteor_score([ref], pred)

def compute_bertscore(pred, ref, model="microsoft/deberta-xlarge-mnli"):
    P, R, F = bertscore([pred], [ref], lang="en", model_type=model)
    return float(F[0])

# --- Reference-free text metrics prompts ---

KEY_ELEMENTS = [
    "limitation",
    "use case",
    "provenance",
    "data collection",
    "evaluation",
    "task",
    "label",
    "intended use",
]

def compute_coverage(pred: str, key_elements: list[str]) -> float:
    coverage = sum(1 for k in key_elements if k.lower() in pred.lower())
    return coverage / len(key_elements)