"""Adapte le reranker multilingue au corpus (livrable L2.4).

Fine-tuning du cross-encoder ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` sur
les paires (question, passage, label) de data/reranker/pairs_train.jsonl, avec
une perte BCE. Le modèle est enregistré dans models/reranker-maintidoc-v1/ avec
une fiche de version (configuration, données, empreinte du jeu d'entraînement).

Usage :  python -m scripts.train_reranker [--epochs 1] [--batch-size 16] [--max-length 320]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

from maintidoc.config import ROOT_DIR
from maintidoc.reranker import ADAPTED_RERANKER, BASE_RERANKER

DATA_DIR = ROOT_DIR / "data" / "reranker"


def load_pairs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ADAPTED_RERANKER))
    args = parser.parse_args()

    import torch
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_file = DATA_DIR / "pairs_train.jsonl"
    pairs = load_pairs(train_file)
    print(f"{len(pairs)} paires d'entraînement ({sum(p['label'] == 1 for p in pairs)} positives).")

    model = CrossEncoder(BASE_RERANKER, num_labels=1, max_length=args.max_length, device="cpu")
    tokenizer, network = model.tokenizer, model.model

    def collate(batch):
        features = tokenizer([b["query"] for b in batch], [b["passage"] for b in batch],
                             padding=True, truncation="only_second", max_length=args.max_length, return_tensors="pt")
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        return features, labels

    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, int(0.1 * total_steps)))
        * max(0.0, (total_steps - step) / total_steps)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    network.train()
    start, step = time.time(), 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for features, labels in loader:
            logits = network(**features).logits.view(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            step += 1
            if step % 10 == 0:
                print(f"époque {epoch} pas {step}/{total_steps} perte {running / 10:.4f}", flush=True)
                running = 0.0
    network.eval()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    card = {
        "name": output.name,
        "version": "1.0.0",
        "created": datetime.now().isoformat(timespec="seconds"),
        "base_model": BASE_RERANKER,
        "task": "reranking question -> passage (cross-encoder, sortie logit)",
        "train_file": str(train_file.relative_to(ROOT_DIR)).replace("\\", "/"),
        "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
        "train_pairs": len(pairs),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "loss": "BCEWithLogits",
        "seed": args.seed,
        "training_seconds": round(time.time() - start, 1),
    }
    (output / "maintidoc_model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    # Copie versionnée dans le dépôt (le modèle lui-même n'est pas poussé sur git).
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / f"{output.name}.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
