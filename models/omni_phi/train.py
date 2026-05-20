import sys
import os
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments

# Resolve paths relative to this script's directory for maximum robustness
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data" / "preprocessed"
OUTPUT_DIR = SCRIPT_DIR / "checkpoints"

# Add models/omni_phi and the project root to sys.path to allow imports from any CWD
omni_phi_dir = str(SCRIPT_DIR)
project_root = os.path.abspath(os.path.join(omni_phi_dir, "..", ".."))
for path in [omni_phi_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

from model    import OmniPhiS2ST
from dataset  import OmniPhiDataset
from collator import omni_phi_collate_fn

MODEL_ID    = "microsoft/Phi-4-multimodal-instruct"

def main():
    print("=== Omni-Phi S2ST Training ===")
    print(f"Model ID  : {MODEL_ID}")
    print(f"Data Dir  : {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print("==============================\n")

    # ── 1. Load model (LoRA adapter applied inside __init__) ────────────────
    model = OmniPhiS2ST(phi4_model_id=MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. Load datasets ────────────────────────────────────────────────────
    train_dataset = OmniPhiDataset(str(DATA_DIR / "train.jsonl"), model.processor, training=True)
    eval_dataset  = OmniPhiDataset(str(DATA_DIR / "eval.jsonl"),  model.processor, training=False)

    # ── 3. Training arguments (mirrors the official Phi-4 script) ───────────
    training_args = TrainingArguments(
        output_dir                   = str(OUTPUT_DIR),
        num_train_epochs             = 1,
        per_device_train_batch_size  = 4,
        gradient_accumulation_steps  = 8,           # effective batch = 32
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs= {"use_reentrant": False},
        optim                        = "adamw_torch",
        adam_beta1                   = 0.9,
        adam_beta2                   = 0.95,
        adam_epsilon                 = 1e-7,
        learning_rate                = 4e-5,
        weight_decay                 = 0.01,
        max_grad_norm                = 1.0,
        lr_scheduler_type            = "linear",
        warmup_steps                 = 50,
        logging_steps                = 10,
        save_strategy                = "epoch",
        bf16                         = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns        = False,       # REQUIRED for custom batch keys
        report_to                    = "none",
        dataloader_num_workers       = 4,
        ddp_find_unused_parameters   = True,        # for frozen SigLIP layers in Phi-4
    )

    # ── 4. Instantiate Trainer ───────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        args          = training_args,
        data_collator = omni_phi_collate_fn,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    model.processor.save_pretrained(str(OUTPUT_DIR))
    print(f"[train.py] Training complete. Checkpoint saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
