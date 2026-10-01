# Frozen DistilBERT sentence embeddings
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embed_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
embed_model.eval()  # Frozen: no fine-tuning, no dropout at inference

@torch.no_grad()
def get_embeddings(texts, batch_size=64, max_length=32, show_progress=True, skip_special=False):
    # skip_special leaves [CLS] and [SEP] out of the mean, they dominate the embedding of very short texts
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(DEVICE)

        output = embed_model(**encoded)
        token_embeddings = output.last_hidden_state

        attention = encoded["attention_mask"].clone()
        if skip_special:
            attention[encoded["input_ids"] == tokenizer.cls_token_id] = 0
            attention[encoded["input_ids"] == tokenizer.sep_token_id] = 0
        mask = attention.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)
