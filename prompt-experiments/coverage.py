import re
import json
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2") 

##---- Coverage Score 
'''


# For the coverage score we will be taking into account the similarity to the key concepts from the reference
This way : iwe are seeing how interpretable the model is being and robust to paraphasing. 

1) Extract the concept units from the reference 
 - split reference into sentences and split the punctuation 
 - Keep chunks
 
2) SPlit the generated description 
3) Embed the reference concepts and the generated sentences
4) Match reference concept to best generated sentence: MAX COSINE SIMILARITY TO ANY GENERATED SENTENCE 
5) compute coverage 
    Haqrd coverage % of reference concepts with best similarity above a threshold
    Soft coverage ave4rgae beat similarity across reference concepts
    

'''
# For the coverage score we will be taking into account the similarity to the key concepts from the reference
    

def _sentences(text: str):
    # simple sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]

def _key_phrases(reference: str, max_phrases: int = 25):
    # lightweight “phrase” extractor: grab longer tokens and some multiword patterns
    ref = re.sub(r"\s+", " ", str(reference)).strip()
    # take candidate phrases as comma/semicolon-separated chunks + sentences
    chunks = re.split(r"[;•\n]", ref)
    cands = []
    for ch in chunks:
        ch = ch.strip()
        if 8 <= len(ch) <= 140:
            cands.append(ch)
    # fallback: use sentences
    if len(cands) < 5:
        cands = _sentences(ref)
    return cands[:max_phrases]

def embedding_coverage(reference: str, generated: str, thresh: float = 0.70) -> float:
    phrases = _key_phrases(reference)
    gen_sents = _sentences(generated)
    if not phrases or not gen_sents:
        return 0.0

    P = _model.encode(phrases, normalize_embeddings=True)
    G = _model.encode(gen_sents, normalize_embeddings=True)

    # cosine sim since normalized => dot product
    sims = P @ G.T
    best = sims.max(axis=1)
    covered = (best >= thresh).mean()
    return float(covered)

def embedding_coverage_mean_sim(reference: str, generated: str) -> float:
    phrases = _key_phrases(reference)
    gen_sents = _sentences(generated)
    if not phrases or not gen_sents:
        return 0.0
    P = _model.encode(phrases, normalize_embeddings=True)
    G = _model.encode(gen_sents, normalize_embeddings=True)
    best = (P @ G.T).max(axis=1)
    return float(np.mean(best))



# -------------------------
# New: keyword coverage
# -------------------------
_STOP = set("""
a an the and or but if then else for to of in on at by with without from as is are was were be been being
this that these those it its into over under between about such than via per
""".split())

def _tokens(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s_\-\/\.]", " ", text)  # keep underscores, slashes, dots for field/unit-ish tokens
    toks = [t for t in text.split() if t and t not in _STOP and len(t) > 2]
    return toks

def keyword_coverage(reference: str, generated: str, top_k: int = 40) -> float:
    ref_toks = _tokens(reference)
    gen_set = set(_tokens(generated))
    if not ref_toks:
        return 0.0
    top = [w for w, _ in Counter(ref_toks).most_common(top_k)]
    hit = sum(1 for w in top if w in gen_set)
    return float(hit / max(len(top), 1))

# -------------------------
# New: numbers / years / units coverage
# -------------------------
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# small, extensible unit/acronym list you can grow
_UNITS = {
    "mg/dl", "mmhg", "kg", "g", "mg", "ml", "l", "hz", "khz",
    "ecg", "eeg", "mri", "ct", "rna-seq", "rnaseq", "scrna", "scrna-seq",
    "hdf5", "csv", "tsv", "fasta", "fastq", "bp", "kb", "mb", "gb",
    "12-lead", "12lead"
}

def _unit_tokens(text: str):
    t = str(text).lower()
    # normalize common variants
    t = t.replace("rna seq", "rna-seq").replace("12 lead", "12-lead")
    toks = set(_tokens(t))
    # also consider raw patterns like mg/dL that might get split strangely
    raw = set(re.findall(r"[a-z]+\/[a-z]+", t))
    return {u for u in (toks | raw) if u in _UNITS}

def _coverage_of_set(ref_items, gen_items) -> float:
    ref_items = set(ref_items)
    gen_items = set(gen_items)
    if not ref_items:
        return 1.0  # if reference has none, don't penalize
    return float(len(ref_items & gen_items) / len(ref_items))

def number_year_unit_coverage(reference: str, generated: str):
    ref_years = set(_YEAR_RE.findall(str(reference)))
    gen_years = set(_YEAR_RE.findall(str(generated)))

    ref_nums = set(_NUM_RE.findall(str(reference)))
    gen_nums = set(_NUM_RE.findall(str(generated)))

    ref_units = _unit_tokens(reference)
    gen_units = _unit_tokens(generated)

    year_cov = _coverage_of_set(ref_years, gen_years)
    num_cov  = _coverage_of_set(ref_nums, gen_nums)
    unit_cov = _coverage_of_set(ref_units, gen_units)

    # combine (years are usually important, but not always present)
    combined = float(np.mean([year_cov, num_cov, unit_cov]))
    return combined, year_cov, num_cov, unit_cov

# -------------------------
# New: field/schema coverage (snake_case, camelCase-ish, quoted identifiers)
# -------------------------
_FIELD_RE = re.compile(r"\b[a-zA-Z]+[a-zA-Z0-9]*_[a-zA-Z0-9_]+\b")  # snake_case
def _field_tokens(text: str):
    t = str(text)
    snake = set(_FIELD_RE.findall(t))
    # also treat backticked/quoted tokens as fields
    quoted = set(re.findall(r"[`'\"]([A-Za-z0-9_]+)[`'\"]", t))
    return {s.lower() for s in (snake | quoted)}

def field_coverage(reference: str, generated: str) -> float:
    ref_fields = _field_tokens(reference)
    gen_fields = _field_tokens(generated)
    return _coverage_of_set(ref_fields, gen_fields)

# -------------------------
# New: dimension coverage (what/who/when/where/how big/what for)
# This is intentionally lightweight heuristics.
# -------------------------
_GEO_CUES = {"country", "state", "province", "city", "region", "county", "europe", "california", "brazil", "spain"}
_USE_CUES = {"predict", "classification", "regression", "analyze", "analysis", "model", "research", "benchmark", "train"}
_POP_CUES = {"patient", "donor", "individual", "species", "participants", "women", "men", "children", "cells"}

def _has_any(text: str, cues) -> bool:
    toks = set(_tokens(text))
    return any(c in toks for c in cues)

def dimension_coverage(reference: str, generated: str) -> float:
    """
    Score dimensions that are present in the reference and also present in generated.
    """
    ref = str(reference)
    gen = str(generated)

    dims = {}

    # when: years or temporal words
    dims["when"] = bool(_YEAR_RE.search(ref) or _has_any(ref, {"year","years","month","daily","weekly","between","from","to"}))
    # where: geo cues or obvious location-like capitalized sequences (very rough)
    dims["where"] = _has_any(ref, _GEO_CUES) or bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", ref))
    # who: population cues
    dims["who"] = _has_any(ref, _POP_CUES)
    # how big: any numbers
    dims["how_big"] = bool(_NUM_RE.search(ref))
    # what for: use-case cues
    dims["what_for"] = _has_any(ref, _USE_CUES)
    # what: assume always present if reference non-empty
    dims["what"] = bool(ref.strip())

    # Only evaluate dims that are present in reference
    present = [k for k, v in dims.items() if v]
    if not present:
        return 1.0

    # Check presence in generated using similar cues
    ok = 0
    for k in present:
        if k == "when":
            ok += int(bool(_YEAR_RE.search(gen) or _has_any(gen, {"year","years","month","daily","weekly","between","from","to"})))
        elif k == "where":
            ok += int(_has_any(gen, _GEO_CUES) or bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", gen)))
        elif k == "who":
            ok += int(_has_any(gen, _POP_CUES))
        elif k == "how_big":
            ok += int(bool(_NUM_RE.search(gen)))
        elif k == "what_for":
            ok += int(_has_any(gen, _USE_CUES))
        elif k == "what":
            ok += int(bool(gen.strip()))
    return float(ok / len(present))

# -------------------------
# New: repetition penalty (0 good -> 1 bad)
# -------------------------
def repetition_penalty(text: str, n: int = 4) -> float:
    toks = _tokens(text)
    if len(toks) < n * 2:
        return 0.0
    ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    if not ngrams:
        return 0.0
    uniq_ratio = len(set(ngrams)) / len(ngrams)
    # penalty higher when uniq_ratio is low
    return float(1.0 - uniq_ratio)

# -------------------------
# New: hallucinated numbers penalty (0 good -> 1 bad)
# -------------------------
def hallucinated_number_penalty(reference: str, generated: str) -> float:
    ref_nums = set(_NUM_RE.findall(str(reference)))
    gen_nums = set(_NUM_RE.findall(str(generated)))
    if not gen_nums:
        return 0.0
    # numbers in generated but not in reference
    extra = gen_nums - ref_nums
    return float(len(extra) / len(gen_nums))

# -------------------------
# Bundle everything + composite
# NOTE: bertscore_recall can be passed in from your earlier BERTScore run.
# -------------------------
def coverage_bundle(
    reference: str,
    generated: str,
    *,
    bertscore_recall: float | None = None,
    concept_thresh: float = 0.70,
    keyword_top_k: int = 40,
):
    reference = "" if reference is None else str(reference)
    generated = "" if generated is None else str(generated)

    cov_concept = embedding_coverage(reference, generated, thresh=concept_thresh)
    cov_concept_soft = embedding_coverage_mean_sim(reference, generated)
    cov_keyword = keyword_coverage(reference, generated, top_k=keyword_top_k)

    cov_num_combo, cov_year, cov_num, cov_unit = number_year_unit_coverage(reference, generated)
    cov_field = field_coverage(reference, generated)
    cov_dim = dimension_coverage(reference, generated)

    pen_rep = repetition_penalty(generated)
    pen_halnum = hallucinated_number_penalty(reference, generated)

    # If you already computed BERTScore recall elsewhere, pass it in.
    cov_bert_r = float(bertscore_recall) if bertscore_recall is not None else None

    # Composite score: use BERTScore recall if available, else fall back to concept_soft
    # (weights are a sensible starting point; tune later)
    core_sem = cov_bert_r if cov_bert_r is not None else cov_concept_soft

    coverage_final = (
        0.30 * cov_concept +
        0.20 * core_sem +
        0.15 * cov_keyword +
        0.15 * cov_num_combo +
        0.10 * cov_field +
        0.10 * cov_dim
        - 0.05 * pen_rep
        - 0.05 * pen_halnum
    )

    # clamp to [0,1]
    coverage_final = float(max(0.0, min(1.0, coverage_final)))

    return {
        "coverage_final": coverage_final,
        "coverage_concept": cov_concept,
        "coverage_concept_soft": cov_concept_soft,
        "coverage_keyword": cov_keyword,
        "coverage_numbers_combo": cov_num_combo,
        "coverage_year": cov_year,
        "coverage_number": cov_num,
        "coverage_unit": cov_unit,
        "coverage_field": cov_field,
        "coverage_dimension": cov_dim,
        "penalty_repetition": pen_rep,
        "penalty_hallucinated_numbers": pen_halnum,
        "coverage_bertscore_r": cov_bert_r,
        "concept_thresh": concept_thresh,
        "keyword_top_k": keyword_top_k,
    }