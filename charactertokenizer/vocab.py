from typing import Dict, List, Optional, Sequence, Tuple, Union, Sequence
from collections import Counter

from datasets import load_dataset

def build_vocab(hf_dataset: Union[str, Tuple], text_column: str = "text", split: str = "train", vocab_limit: int = None) -> Sequence[str]:

    if isinstance(hf_dataset, str):
        hf_dataset = (hf_dataset,)

    dataset = load_dataset(*hf_dataset, split=split)

    counter = Counter()
    
    total = len(dataset)
    for i, row in enumerate(dataset):
        print(f"\rProcessing row {i}/{total}", end="")
        counter.update(row[text_column])

    vocab = []
    for i, (char, count) in enumerate(counter.most_common(vocab_limit)):
        vocab.append(char)

    return vocab

