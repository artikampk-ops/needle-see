"""Train the visual graft: LoRA on the attention stack + the visual projector.

The base checkpoint is frozen; gradients flow through

    CLIP patch features -> projector -> VisualSAN (LoRA-adapted) -> tool call

Loss is the exact same masked cross-entropy the stock local fine-tune uses,
with the visual prefix excluded from the loss (it predicts no tokens).

    .venv/bin/python -m needlesee.train --smoke                 # local sanity run
    .venv/bin/python -m needlesee.train --data-dir out/data --features out/features \
        --layers 20 --epochs 6 --batch-size 32                  # the real run

Artifacts land in ``--out`` (default ``out/graft``) in the graft bundle format
(see :mod:`needlesee.graft`).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..paths import checkpoint_path, ensure_needle

ensure_needle()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from needle.model.finetune import (  # noqa: E402
    init_lora,
    lora_target_paths,
    merge_lora,
    render_example,
    rung,
)
from needle.model.run import load_checkpoint  # noqa: E402
from needle.model.tokenizer import BOS_ID, get_tokenizer  # noqa: E402

from ..graft import save_bundle  # noqa: E402
from ..model3 import VisualSAN  # noqa: E402
from ..projector import VisualProjector  # noqa: E402
from ..runtime.decoding import encode_example, score_completions  # noqa: E402


def read_split(jsonl_path: Path, limit: int | None = None):
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def load_features(features_dir: Path, split: str, limit: int | None):
    candidate = features_dir / f"{split}.npy"
    if candidate.is_file():
        features = np.load(candidate, mmap_mode="r")
        if limit:
            features = features[:limit]
        return features
    raise FileNotFoundError(
        f"no features for split '{split}' under {features_dir}; run "
        f"`python -m needlesee.data.precompute --jsonl {features_dir.parent/'data'}/{split}.jsonl "
        f"--out {features_dir}` first"
    )


def evaluate_candidates(model, params, projector, projector_params, tokenizer, rows,
                        features, n: int, rung_layers: int) -> dict:
    correct = defaultdict(int)
    total = defaultdict(int)
    for index in range(min(n, len(rows))):
        row = rows[index]
        prompt, _ = render_example({**row, "answers": []})
        prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
        scores = score_completions(model, params, projector, projector_params,
                                   features[index], tokenizer, prompt_ids,
                                   row["candidates"])
        predicted = int(np.argmax(scores))
        task = row["task"]
        total[task] += 1
        correct[task] += int(predicted == int(row["truth_index"]))
    overall = (sum(correct.values()), sum(total.values()))
    return {
        "per_task": {task: {"correct": correct[task], "total": total[task],
                            "accuracy": correct[task] / max(total[task], 1)}
                     for task in sorted(total)},
        "overall": {"correct": overall[0], "total": overall[1],
                    "accuracy": overall[0] / max(overall[1], 1)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="train the visual graft")
    parser.add_argument("--data-dir", default="out/data")
    parser.add_argument("--features", default="out/features")
    parser.add_argument("--out", default="out/graft")
    parser.add_argument("--checkpoint", default=None, help="base .safetensors override")
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=3e-4, help="LoRA learning rate")
    parser.add_argument("--projector-lr", type=float, default=1e-3)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="cap train rows (0 = all)")
    parser.add_argument("--eval-val", type=int, default=64, help="candidate-scoring val examples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU sanity run: 2 layers, 96 rows, 6 epochs")
    args = parser.parse_args()

    if args.smoke:
        args.layers = 2
        args.limit = args.limit or 96
        args.epochs = args.epochs if args.epochs != 6 else 8
        args.batch_size = min(args.batch_size, 8)
        args.eval_val = min(args.eval_val, 32)

    started = time.time()
    base_path = args.checkpoint or str(checkpoint_path())
    params, config = load_checkpoint(base_path)
    config.dtype = args.dtype
    params = jax.tree.map(lambda a: np.asarray(a).astype(args.dtype), params)
    params, config = rung(params, config, args.layers)
    params = jax.device_put(params)
    tokenizer = get_tokenizer(config.vocab_size)
    model = VisualSAN(config)
    print(f"  {'base':<10} {base_path}")
    print(f"  {'depth':<10} {config.num_layers} layers  dtype {args.dtype}")

    data_dir = Path(args.data_dir)
    train_rows = read_split(data_dir / "train.jsonl", args.limit or None)
    val_rows = read_split(data_dir / "val.jsonl", 512)
    train_features = load_features(Path(args.features), "train", args.limit or None)
    val_features = load_features(Path(args.features), "val", 512)
    if len(train_rows) != len(train_features):
        raise SystemExit("train.jsonl and train features disagree in length")
    if len(val_rows) != len(val_features):
        raise SystemExit("val.jsonl and val features disagree in length")
    print(f"  {'data':<10} {len(train_rows)} train  {len(val_rows)} val  "
          f"seq_len {args.max_len}")

    t0 = time.time()
    train_ids, train_masks = [], []
    for row in train_rows:
        ids, mask = encode_example(tokenizer, row, args.max_len)
        train_ids.append(ids)
        train_masks.append(mask)
    val_ids, val_masks = [], []
    for row in val_rows:
        ids, mask = encode_example(tokenizer, row, args.max_len)
        val_ids.append(ids)
        val_masks.append(mask)
    train_ids = np.stack(train_ids)
    train_masks = np.stack(train_masks)
    val_ids = np.stack(val_ids)
    val_masks = np.stack(val_masks)
    print(f"  {'encode':<10} {time.time() - t0:.1f}s")

    feature_dim = int(train_features.shape[2])
    feature_tokens = int(train_features.shape[1])
    projector = VisualProjector(d_model=config.d_model, num_tokens=feature_tokens,
                                feature_dim=feature_dim, hidden=args.projector_hidden)
    key = jax.random.PRNGKey(args.seed)
    projector_params = projector.init(key, jnp.zeros((1, feature_tokens, feature_dim)))
    lora_paths = lora_target_paths(params)
    lora = init_lora(params, lora_paths, args.lora_rank, jax.random.PRNGKey(args.seed + 1))
    scale = args.lora_alpha / args.lora_rank
    print(f"  {'graft':<10} lora rank {args.lora_rank} x{len(lora_paths)} groups  "
          f"projector {feature_dim}->{args.projector_hidden}->{config.d_model}  "
          f"{feature_tokens} tokens")

    count = len(train_rows)
    batch = args.batch_size
    steps_per_epoch = -(-count // batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(max(1, total_steps // 20), max(total_steps - 1, 1))

    def schedule(peak):
        return optax.warmup_cosine_decay_schedule(init_value=peak * 0.05,
                                                  peak_value=peak,
                                                  warmup_steps=warmup,
                                                  decay_steps=total_steps)

    variables = {"lora": lora, "proj": projector_params["params"]}
    labels = {"lora": jax.tree.map(lambda _: "lora", lora),
              "proj": jax.tree.map(lambda _: "proj", projector_params["params"])}
    optimizer = optax.multi_transform(
        {
            "lora": optax.chain(optax.clip_by_global_norm(1.0),
                                optax.adamw(schedule(args.lr))),
            "proj": optax.chain(optax.clip_by_global_norm(1.0),
                                optax.adamw(schedule(args.projector_lr))),
        },
        labels,
    )
    opt_state = optimizer.init(variables)
    print(f"  {'schedule':<10} {total_steps} steps  warmup {warmup}  "
          f"lr {args.lr:g}/{args.projector_lr:g}  (compiling...)")

    def loss_fn(variables, ids, vis, mask):
        merged = merge_lora(params, variables["lora"], scale)
        v = projector.apply({"params": variables["proj"]}, vis)
        logits = model.apply({"params": merged}, ids, visual_embeds=v)
        prefix = v.shape[1]
        logits = logits[:, prefix - 1:prefix - 1 + ids.shape[1]]
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    @jax.jit
    def train_step(variables, opt_state, ids, vis, mask):
        loss, grads = jax.value_and_grad(loss_fn)(variables, ids, vis, mask)
        updates, opt_state = optimizer.update(grads, opt_state, variables)
        return optax.apply_updates(variables, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    rng = np.random.default_rng(args.seed)
    every = max(1, total_steps // 40)
    step = 0
    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        last = 0.0
        for start in range(0, count, batch):
            index = order[start:start + batch]
            variables, opt_state, loss = train_step(
                variables, opt_state,
                jnp.asarray(train_ids[index]),
                jnp.asarray(train_features[index], jnp.float32),
                jnp.asarray(train_masks[index]))
            last = float(loss)
            step += 1
            if step % every == 0 or step == total_steps:
                print(f"  {'step':<10} {step}/{total_steps}  loss {last:.4f}  "
                      f"{(time.time() - started) / 60:.1f} min")
        summary = []
        for start in range(0, len(val_rows), batch):
            chunk = slice(start, start + batch)
            val_loss = float(eval_step(variables, jnp.asarray(val_ids[chunk]),
                                       jnp.asarray(val_features[chunk], jnp.float32),
                                       jnp.asarray(val_masks[chunk])))
            summary.append(val_loss)
        val_loss = float(np.mean(summary))
        history.append({"epoch": epoch + 1, "train": last, "val": val_loss})
        print(f"  {'epoch':<10} {epoch + 1}/{args.epochs}  train {last:.4f}  "
              f"val {val_loss:.4f}")

    eval_report = None
    if args.eval_val:
        print(f"  {'eval':<10} candidate scoring on {min(args.eval_val, len(val_rows))} "
              f"val examples ...")
        variables = jax.tree.map(lambda a: jax.device_get(a), variables)
        eval_report = evaluate_candidates(model, params, projector, variables["proj"],
                                          tokenizer, val_rows, val_features,
                                          args.eval_val, config.num_layers)
        overall = eval_report["overall"]
        print(f"  {'accuracy':<10} {overall['correct']}/{overall['total']} exact "
              f"({overall['accuracy'] * 100:.1f}%)")
        for task, stats in eval_report["per_task"].items():
            print(f"             {task:<16} {stats['correct']}/{stats['total']}  "
                  f"{stats['accuracy'] * 100:.0f}%")

    meta = save_bundle(
        args.out,
        lora=jax.tree.map(lambda a: np.asarray(jax.device_get(a)), variables["lora"]),
        scale=scale, rank=args.lora_rank, seed=args.seed, base=base_path,
        layers=config.num_layers,
        projector_params=jax.tree.map(lambda a: np.asarray(jax.device_get(a)),
                                      variables["proj"]),
        projector=projector,
        stats={
            "train_examples": count,
            "val_examples": len(val_rows),
            "epochs": args.epochs,
            "batch_size": batch,
            "max_len": args.max_len,
            "dtype": args.dtype,
            "history": history,
            "eval": eval_report,
            "runtime_minutes": round((time.time() - started) / 60, 2),
        })
    print(f"  {'saved':<10} {args.out}  (lora.safetensors, projector.safetensors, graft.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
