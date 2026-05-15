#!/usr/bin/env python3
"""
Build FAISS index + ID mapping for MS MARCO passages using Contriever.

Outputs:
  artifacts/indexes/msmarco_contriever.index   (FAISS IndexFlatIP)
  artifacts/indexes/msmarco_ids_fixed.npy      (doc ID array)

Usage:
  python build_index.py                        # full corpus (~8.8M passages)
  python build_index.py --max_docs 100000      # quick test subset
  python build_index.py --batch_size 256       # adjust for your GPU/RAM
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import ir_datasets
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INDEX_PATH = "artifacts/indexes/msmarco_contriever.index"
ID_PATH = "artifacts/indexes/msmarco_ids_fixed.npy"


def embed_batch(tokenizer, model, texts, max_length=256):
    """Mean-pooled Contriever embeddings for a batch of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    ).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for MS MARCO passages")
    parser.add_argument("--model_name", default="facebook/contriever-msmarco")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Limit number of passages (for testing). None = full corpus.")
    parser.add_argument("--index_path", default=INDEX_PATH)
    parser.add_argument("--id_path", default=ID_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(DEVICE).eval()

    # Determine embedding dimension from a dummy forward pass
    dummy = embed_batch(tokenizer, model, ["hello"])
    dim = dummy.shape[1]
    print(f"Embedding dim: {dim}")

    # Inner product index (Contriever uses cosine similarity on L2-normalized vectors)
    index = faiss.IndexFlatIP(dim)

    print("Loading MS MARCO passages via ir_datasets ...")
    dataset = ir_datasets.load("msmarco-passage")

    doc_ids = []
    batch_texts = []
    total = 0

    for doc in tqdm(dataset.docs_iter(), desc="Encoding passages"):
        if args.max_docs is not None and total >= args.max_docs:
            break

        doc_ids.append(str(doc.doc_id))
        batch_texts.append(doc.text or "")
        total += 1

        if len(batch_texts) >= args.batch_size:
            vecs = embed_batch(tokenizer, model, batch_texts)
            index.add(vecs)
            batch_texts = []

    # flush remaining
    if batch_texts:
        vecs = embed_batch(tokenizer, model, batch_texts)
        index.add(vecs)

    print(f"\nIndexed {index.ntotal} passages (dim={dim})")

    # Save
    faiss.write_index(index, args.index_path)
    print(f"Saved FAISS index: {args.index_path}")

    np.save(args.id_path, np.array(doc_ids))
    print(f"Saved ID mapping:  {args.id_path}")

    print("Done.")


if __name__ == "__main__":
    main()
