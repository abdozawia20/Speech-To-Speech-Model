"""
lang_config.py
──────────────
Single source-of-truth for all supported target languages in the OmniPhi
Speech-to-Speech Translation pipeline.

Each entry maps a BCP-47 language prefix → a dict containing:
  - name:         Full English name of the language, used to build the
                  Phi-4 prompt: "Translate this to spoken {name}:"
  - whisper_lang: Whisper language tag passed to the ASR pipeline during
                  evaluation.

Adding a new target language
────────────────────────────
1. Add an entry to LANG_CONFIG below.
2. Preprocess data:  python preprocess_omni.py --lang_tgt <prefix>
3. Train:            python train.py --lang_tgt <prefix>
4. Evaluate:         python evaluate_omni_phi.py --lang_tgt <prefix>
5. Infer:            python inference.py --lang_tgt <prefix> --input audio.wav
"""

LANG_CONFIG: dict[str, dict[str, str]] = {
    "de": {
        "name":         "german",
        "whisper_lang": "de",
    },
    "fr": {
        "name":         "french",
        "whisper_lang": "fr",
    },
    "it": {
        "name":         "italian",
        "whisper_lang": "it",
    },
    # ── Add new languages below ────────────────────────────────────────────────
    # "es": {
    #     "name":         "spanish",
    #     "whisper_lang": "es",
    # },
    # "pt": {
    #     "name":         "portuguese",
    #     "whisper_lang": "pt",
    # },
}


def get_lang_name(lang_prefix: str) -> str:
    """Return the full English language name for a given prefix.

    Raises:
        KeyError: if lang_prefix is not in LANG_CONFIG.
    """
    if lang_prefix not in LANG_CONFIG:
        supported = ", ".join(sorted(LANG_CONFIG.keys()))
        raise KeyError(
            f"[lang_config] Unsupported lang_prefix '{lang_prefix}'. "
            f"Supported languages: {supported}. "
            f"Add a new entry to LANG_CONFIG in lang_config.py to enable it."
        )
    return LANG_CONFIG[lang_prefix]["name"]


def get_whisper_lang(lang_prefix: str) -> str:
    """Return the Whisper language tag for a given prefix.

    Raises:
        KeyError: if lang_prefix is not in LANG_CONFIG.
    """
    if lang_prefix not in LANG_CONFIG:
        supported = ", ".join(sorted(LANG_CONFIG.keys()))
        raise KeyError(
            f"[lang_config] Unsupported lang_prefix '{lang_prefix}'. "
            f"Supported languages: {supported}."
        )
    return LANG_CONFIG[lang_prefix]["whisper_lang"]
