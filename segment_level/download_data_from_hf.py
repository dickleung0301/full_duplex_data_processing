from datasets import Dataset, load_dataset

def download_forced_aligned_librispeech(split: str) -> Dataset:
    """
    a function to download forced aligned librispeech
    """

    return load_dataset("gilkeyio/librispeech-alignments", split=split)

def download_huggingface_librispeech(split: str) -> Dataset:
    """
    a function to download librispeech
    """

    return load_dataset("openslr/librispeech_asr", split=split)