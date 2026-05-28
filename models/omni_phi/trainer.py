"""
omni_phi/trainer.py
───────────────────
Custom HF Trainer for OmniPhiS2ST.

Problem
-------
Phi-4-multimodal-instruct (like most LLMs) ties `embed_tokens.weight` and
`lm_head.weight` to the same underlying tensor.  The default HF Trainer uses
`safe_serialization=True` (safetensors format) which refuses to serialize
shared-memory tensors, raising:

    RuntimeError: Some tensors share memory … [{'phi4.model.embed_tokens.weight',
    'phi4.lm_head.weight'}]

Fix
---
Override `save_model()` (called at the end of training) and `_save_checkpoint()`
(called mid-training if save_strategy != "no") to pass `safe_serialization=False`,
which falls back to the standard PyTorch pickle format and handles tied weights
correctly.
"""

import os
from transformers import Trainer


class OmniPhiTrainer(Trainer):
    """
    Drop-in replacement for HF Trainer that handles Phi-4's tied embedding
    weights when saving.

    Usage
    -----
    Replace `Trainer(...)` with `OmniPhiTrainer(...)` in your training cell.
    Everything else stays the same.
    """

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """
        Save the model using pickle-based serialization (safe_serialization=False)
        to correctly handle shared/tied weight tensors.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Unwrap the OmniPhiS2ST wrapper and save the underlying Phi-4 model.
        # `safe_serialization=False` → pytorch_model.bin instead of model.safetensors,
        # which handles tied weights (embed_tokens ↔ lm_head) without errors.
        phi4_model = self.model.phi4
        phi4_model.save_pretrained(output_dir, safe_serialization=False)

        # Save processor / tokenizer if available.
        # NOTE: Phi4MMProcessor.save_pretrained() crashes in transformers 4.57.3
        # with AttributeError: 'audio_tokenizer'. Work around by saving the
        # tokenizer sub-component directly — it contains all vocab/config needed
        # for inference; audio processor parts load fine from the hub.
        processor = None
        if self.processing_class is not None:
            processor = self.processing_class
        elif hasattr(self.model, "processor"):
            processor = self.model.processor

        if processor is not None:
            try:
                processor.save_pretrained(output_dir)
            except AttributeError:
                # Phi4MMProcessor bug: falls back to tokenizer-only save
                if hasattr(processor, "tokenizer"):
                    processor.tokenizer.save_pretrained(output_dir)
                    print("[OmniPhiTrainer] Saved tokenizer (processor.save_pretrained skipped due to Phi4MM bug).")

        # Write a complete preprocessor_config.json from the hub so that
        # AutoProcessor.from_pretrained(output_dir) works on the next load
        # without needing the hub-fallback in model.py.
        try:
            from transformers import AutoProcessor as _AP
            PHI4_HUB_ID = "microsoft/Phi-4-multimodal-instruct"
            _AP.from_pretrained(PHI4_HUB_ID, trust_remote_code=True).save_pretrained(output_dir)
            print("[OmniPhiTrainer] Wrote complete preprocessor_config.json from hub to checkpoint.")
        except Exception as hub_err:
            print(f"[OmniPhiTrainer] Could not write hub processor to checkpoint ({hub_err}). "
                  "model.py fallback will handle future loads.")

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Override mid-training checkpoint saving to also use safe_serialization=False.
        Only relevant when save_strategy = "epoch" or "steps".
        """
        # Let the parent handle all the bookkeeping (optimizer state, scheduler, etc.)
        # then patch the model save that it does internally.
        _orig = model.phi4.save_pretrained

        def _patched_save(save_dir, **kwargs):
            kwargs["safe_serialization"] = False
            return _orig(save_dir, **kwargs)

        model.phi4.save_pretrained = _patched_save
        try:
            super()._save_checkpoint(model, trial, metrics=metrics)
        finally:
            model.phi4.save_pretrained = _orig  # always restore
