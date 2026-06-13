"""
omni_phi/trainer.py
───────────────────
Custom HF Trainer for OmniPhiS2ST.

Problems fixed
--------------
1. Phi-4 ties `embed_tokens.weight` ↔ `lm_head.weight` to the same tensor.
   safetensors refuses to serialize shared-memory tensors → RuntimeError.
   Fix: always pass safe_serialization=False.

2. `save_strategy = "no"` + no `resume_from_checkpoint` → kernel restart loses
   all training progress.
   Fix: `_save_checkpoint` is now correctly implemented so that step-based
   checkpoints (checkpoint-NNN/) are written with full trainer state
   (optimizer.pt, scheduler.pt, trainer_state.json, rng_state.pth).

3. The old `_save_checkpoint` patched `model.phi4.save_pretrained` but the HF
   Trainer 4.57.x calls `self._save_model(output_dir, state_dict=state_dict)`
   which injects the state_dict and uses a completely different code path,
   bypassing the patch.
   Fix: override `_save_model` instead — that is the single, authoritative
   entry-point that both `save_model()` and `_save_checkpoint()` funnel through.
"""

import os
import torch
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class OmniPhiTrainer(Trainer):
    """
    Drop-in replacement for HF Trainer that:
      • handles Phi-4's tied embedding weights (safe_serialization=False)
      • correctly saves/restores full trainer state for mid-training resumption
        (optimizer, scheduler, RNG, step counter, trainer_state.json)

    Usage
    -----
    Replace `Trainer(...)` with `OmniPhiTrainer(...)` in your training cell.
    Pass `resume_from_checkpoint=True` (or a path) to `trainer.train()` to
    automatically resume from the latest checkpoint-NNN/ directory.
    """

    # ── Core save hook ────────────────────────────────────────────────────────
    def _save_model(self, output_dir: str, state_dict=None):
        """
        Single authoritative model-save entry-point.

        Called by BOTH:
          • save_model()          (final save after training)
          • _save_checkpoint()    (mid-training, inside checkpoint-NNN/)

        We unwrap OmniPhiS2ST → phi4 and call save_pretrained with
        safe_serialization=False to avoid the tied-weight RuntimeError.
        The `state_dict` argument (injected by _save_checkpoint in newer
        transformers) is forwarded so sharded saving works correctly.
        """
        os.makedirs(output_dir, exist_ok=True)
        phi4 = self.model.phi4

        save_kwargs = dict(safe_serialization=False)
        if state_dict is not None:
            save_kwargs["state_dict"] = state_dict

        phi4.save_pretrained(output_dir, **save_kwargs)
        print(f"[OmniPhiTrainer] Model weights saved to {output_dir}")

    # ── Public save_model (final / manual saves) ──────────────────────────────
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """
        Save model weights + processor to `output_dir`.
        Called explicitly at the end of training and by trainer.save_model().
        """
        output_dir = output_dir or self.args.output_dir
        self._save_model(output_dir)
        self._save_processor(output_dir)

    # ── Mid-training checkpoint (checkpoint-NNN/) ─────────────────────────────
    def _save_checkpoint(self, model, trial, **kwargs):
        """
        Saves a full resumable checkpoint to output_dir/checkpoint-{global_step}/.

        The directory contains:
          • pytorch_model*.bin   — model weights  (via _save_model above)
          • optimizer.pt         — optimizer state  ┐
          • scheduler.pt         — LR scheduler     │ written by
          • rng_state*.pth       — RNG state        │ super()._save_checkpoint
          • trainer_state.json   — step/epoch/loss  ┘

        **kwargs absorbs any signature changes across transformers versions
        (e.g. `metrics` was removed in 4.57.x).
        """
        super()._save_checkpoint(model, trial, **kwargs)

    # ── Processor / tokenizer helper ──────────────────────────────────────────
    def _save_processor(self, output_dir: str):
        """
        Save processor or fall back to tokenizer-only save.

        Phi4MMProcessor.save_pretrained() crashes in transformers 4.57.3
        with AttributeError: 'audio_tokenizer'. We silently fall back to
        saving the tokenizer sub-component.
        """
        processor = (
            self.processing_class
            if self.processing_class is not None
            else getattr(self.model, "processor", None)
        )
        if processor is None:
            return

        try:
            processor.save_pretrained(output_dir)
        except AttributeError:
            if hasattr(processor, "tokenizer"):
                processor.tokenizer.save_pretrained(output_dir)
                print("[OmniPhiTrainer] Saved tokenizer only "
                      "(processor.save_pretrained skipped — Phi4MM bug).")
