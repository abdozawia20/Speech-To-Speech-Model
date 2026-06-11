import sys
import os
import argparse
from pathlib import Path
import torch
from transformers import TrainingArguments

# Resolve paths relative to this script's directory for maximum robustness
SCRIPT_DIR = Path(__file__).resolve().parent

# Add models/omni_phi and the project root to sys.path to allow imports from any CWD
omni_phi_dir = str(SCRIPT_DIR)
project_root = os.path.abspath(os.path.join(omni_phi_dir, "..", ".."))
for path in [omni_phi_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

from model    import OmniPhiS2ST
from dataset  import OmniPhiDataset
from collator import omni_phi_collate_fn
from trainer  import OmniPhiTrainer

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OmniPhiS2ST for a given target language."
    )
    parser.add_argument(
        "--lang_tgt",
        type=str,
        default="de",
        help="Target language ISO prefix, e.g. 'de', 'fr', 'it' (default: 'de'). "
             "Must match an entry in lang_config.LANG_CONFIG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lang_tgt = args.lang_tgt

    # ── Derive language-namespaced directories ────────────────────────────────
    DATA_DIR   = SCRIPT_DIR / "data" / f"preprocessed_en_{lang_tgt}"
    OUTPUT_DIR = SCRIPT_DIR / f"checkpoints_en_{lang_tgt}"

    print("=== Omni-Phi S2ST Training ===")
    print(f"Model ID   : {MODEL_ID}")
    print(f"Language   : en -> {lang_tgt}")
    print(f"Data Dir   : {DATA_DIR}")
    print(f"Output Dir : {OUTPUT_DIR}")
    print("==============================\n")

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print(
            f"[train.py] ERROR: Data directory '{DATA_DIR}' is empty or does not exist.\n"
            f"           Run: python preprocess_omni.py --lang_tgt {lang_tgt} --split train\n"
            f"           then: python preprocess_omni.py --lang_tgt {lang_tgt} --split eval"
        )
        sys.exit(1)

    # ── 1. Load model (LoRA adapter applied inside __init__) ──────────────────
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = OmniPhiS2ST(phi4_model_id=MODEL_ID, device=device, lang_prefix=lang_tgt)

    # ── 2. Load datasets ──────────────────────────────────────────────────────
    train_dataset = OmniPhiDataset(
        str(DATA_DIR / "train.jsonl"), model.processor,
        training=True, lang_prefix=lang_tgt,
    )
    eval_jsonl_path = DATA_DIR / "eval.jsonl"
    if eval_jsonl_path.exists():
        eval_dataset = OmniPhiDataset(
            str(eval_jsonl_path), model.processor,
            training=False, lang_prefix=lang_tgt,
        )
    else:
        print(f"[train.py] {eval_jsonl_path} not found. Skipping evaluation dataset.")
        eval_dataset = None

    # ── 3. Training arguments (mirrors the official Phi-4 script) ─────────────
    # ── Detect if fused AdamW is supported (requires CUDA + torch >= 2.0) ─────
    use_fused_adam = (
        torch.cuda.is_available()
        and hasattr(torch.optim, "AdamW")
        and "fused" in torch.optim.AdamW.__init__.__doc__
    ) if torch.cuda.is_available() else False
    optim_choice = "adamw_torch_fused" if use_fused_adam else "adamw_torch"
    print(f"[train.py] Using optimizer: {optim_choice}")

    training_args = TrainingArguments(
        output_dir                   = str(OUTPUT_DIR),
        num_train_epochs             = 5,
        # max_steps                  = 2,           # REMOVED: was a debug cap
        per_device_train_batch_size  = 2,            # A100 80GB can handle batch=2 with LoRA
        gradient_accumulation_steps  = 8,            # effective batch = 16; larger = better GPU util
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs= {"use_reentrant": False},
        optim                        = optim_choice, # fused AdamW: ~2x faster kernel launches
        adam_beta1                   = 0.9,
        adam_beta2                   = 0.95,
        adam_epsilon                 = 1e-7,
        learning_rate                = 4e-5,
        weight_decay                 = 0.01,
        max_grad_norm                = 1.0,
        lr_scheduler_type            = "cosine",     # cosine > linear for LoRA fine-tuning
        warmup_ratio                 = 0.03,         # 3% warmup, scales with total steps
        logging_steps                = 5,
        save_strategy                = "epoch",
        save_safetensors             = False,         # CRITICAL: prevents tied-weight RuntimeError
        bf16                         = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32                         = True,          # enable TF32 matmuls on A100 (free speedup)
        remove_unused_columns        = False,        # REQUIRED for custom batch keys
        report_to                    = "none",
        dataloader_num_workers       = 2,            # 2 workers; heavy CPU processing is pre-cached
        dataloader_pin_memory        = True,         # pinned memory → faster CPU→GPU transfers
        ddp_find_unused_parameters   = False,        # LoRA only trains a subset; set True only for DDP multi-GPU
    )

    # ── 4. Instantiate Trainer ────────────────────────────────────────────────
    trainer = OmniPhiTrainer(
        model         = model,
        args          = training_args,
        data_collator = omni_phi_collate_fn,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    # Save processor (Phi4MMProcessor.save_pretrained crashes in transformers 4.57.3
    # with AttributeError: 'audio_tokenizer'. Fall back to tokenizer-only save.)
    try:
        model.processor.save_pretrained(str(OUTPUT_DIR))
    except AttributeError:
        if hasattr(model.processor, "tokenizer"):
            model.processor.tokenizer.save_pretrained(str(OUTPUT_DIR))
            print("[train.py] Saved tokenizer (processor.save_pretrained skipped due to Phi4MM bug).")
    print(f"[train.py] Training complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
