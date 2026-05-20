from dataset_analysis.validator import FleursSemanticValidator

def test_normalization():
    # Force CPU and tiny model for fast unit testing
    validator = FleursSemanticValidator(model_size='tiny', device='cpu')
    text = "Hello, World!  This is a TEST."
    expected = "hello world this is a test"
    actual = validator.normalize_text(text)
    print(f"  Normalization: '{text}' -> '{actual}'")
    assert actual == expected

def test_wer_calculation():
    validator = FleursSemanticValidator(model_size='tiny', device='cpu')
    ref = "the quick brown fox"
    hyp = "the quick brown dog"
    # 1 error / 4 words = 0.25
    wer = validator.calculate_wer(ref, hyp)
    print(f"  WER Calculation: '{ref}' vs '{hyp}' -> {wer}")
    assert wer == 0.25

if __name__ == "__main__":
    # Simple manual run
    try:
        print("Starting basic tests...")
        test_normalization()
        test_wer_calculation()
        print("All basic tests passed.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
