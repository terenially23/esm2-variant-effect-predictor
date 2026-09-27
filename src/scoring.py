"""Shared ESM2 zero-shot variant scoring logic, used by both the offline CLI
script (src/score_variants.py) and the Lambda handler (src/lambda_handler.py)."""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MAX_CONTEXT = 1022  # ESM2's max_position_embeddings (1024) minus <cls>/<eos>

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).eval()
    return _tokenizer, _model


def truncate_around_position(seq: str, pos: int) -> tuple[str, int]:
    """Keeps the variant position inside the model's context window for long
    proteins by windowing around it rather than truncating from one end."""
    if len(seq) <= MAX_CONTEXT:
        return seq, pos
    half = MAX_CONTEXT // 2
    start = max(0, pos - 1 - half)
    end = min(len(seq), start + MAX_CONTEXT)
    start = max(0, end - MAX_CONTEXT)
    return seq[start:end], pos - start


def score_variant(sequence: str, pos: int, wt: str, mt: str) -> float:
    """Masked-marginal zero-shot score: log P(mutant) - log P(wild-type) at the
    variant position, given the rest of the sequence. More negative = the model
    finds the substitution less likely, i.e. more likely to disrupt function."""
    tokenizer, model = load_model()

    windowed_seq, windowed_pos = truncate_around_position(sequence, pos)
    inputs = tokenizer(windowed_seq, return_tensors="pt")
    token_index = windowed_pos  # +1 for <cls> is already implicit in this index

    actual = tokenizer.convert_ids_to_tokens([inputs["input_ids"][0, token_index].item()])[0]
    if actual != wt:
        raise ValueError(f"Expected wild-type residue {wt} at position {pos}, sequence has {actual}")

    masked_input_ids = inputs["input_ids"].clone()
    masked_input_ids[0, token_index] = tokenizer.mask_token_id

    with torch.no_grad():
        logits = model(input_ids=masked_input_ids, attention_mask=inputs["attention_mask"]).logits
    log_probs = torch.log_softmax(logits[0, token_index], dim=-1)

    wt_id = tokenizer.convert_tokens_to_ids(wt)
    mt_id = tokenizer.convert_tokens_to_ids(mt)
    return (log_probs[mt_id] - log_probs[wt_id]).item()
