from itertools import chain
import logging

from datasets import load_dataset
import nltk
from overlapy import OverlapyTestSet, Overlapy


# We are going to search for overlaps in a parallel manner, meaning each
# Process is going to potentially load the dataset

logging.getLogger("datasets.builder").setLevel(logging.ERROR)

nltk.download("punkt")


def tokenizer(s):
    # Tokenizer used in https://arxiv.org/abs/2005.14165
    # Tokenize by whitespace, allow only alphanumeric characters
    # And lowercase all words
    return [word.lower() for word in nltk.word_tokenize(s) if word.isalnum()]


# The next three functions transform a dataset example into sequences of tokens.
# This example performs a data contamination study on the the following task:
#      * Physical Interaction: Question Answering (PIQA) - Bisk, Yonatan, et al. "Piqa: Reasoning about physical commonsense in natural language." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 05. 2020.

# Datasets can have several input segments. As an example, PIQA has 1 Goal and 2 candidate solutions.
# As such we tokenize the 2 possible pairings, as we can have 2 possible overlaps:
#    1. Goal + Solution 1
#    2. Goal + Solution 2


def piqa_example_to_tokens(example):
    tokens = []
    for i in range(1, 3):
        tokens.append(tokenizer(example["goal"]) + tokenizer(example[f"sol{i}"]))
    return tokens


# We load the dataset, turn it into sequences of tokens using our example_to_tokens function
# And then create an OverlapyTestSet object

piqa = load_dataset("piqa")
examples = list(
    chain(*[piqa_example_to_tokens(example) for example in piqa["validation"]])
)
piqa_testset = OverlapyTestSet("piqa", examples=examples)
print("PIQA Testset")
print(f"N value: {piqa_testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(piqa_testset.ngrams()))))}")


# We're defining a wrapper for a HuggingFace dataset to make it compatible with overlapy.
# Overlapy expects __getitem__ to return the example's text as ngrams.
# However, HuggingFace Dataset's __getitem__ returns a dictionary, and the text is
# Stored in the key "text". As such, we define a wrapper that receives a HF Dataset
# And the __getitem__ receives an idx, accesses the dataset idx and selects the "text" key value
# and tokenizes it

class HuggingFaceDatasetWrapper:
    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.tokenizer(self.ds[idx]["text"])

    def __len__(self):
        return len(self.ds)


# We are analyzing overlaps with OpenWebText (https://skylion007.github.io/OpenWebTextCorpus/)
# This language modeling dataset was used in models such as the GPT series and RoBERTa.
dataset = load_dataset("openwebtext")["train"]

# We create an Overlapy object, handing three arguments:
#   * Testsets: A list of OverlapyTestSet objects that we want to study.
#   * Dataset: Dataset we want to calculate collisions with
#   * n_workers: Number of worker processes to use
# It's advisable to stay below 32 workers for HuggingFace datasets.
# In our experience, more than that leads to race conditions and
# it permanently stops.
overlapy = Overlapy(
    testsets=[piqa_testset],
    dataset=HuggingFaceDatasetWrapper(dataset, tokenizer),
    n_workers=32,
)
# Let's run and get the matches
matches = overlapy.run()

# Using the matches dictionary, we use the OverlapyTestSet object for PIQA
# To obtain information about which examples were found to have matches with
# OpenWebText

# The output follows the structure: Example ID, Ngram, Match position within example sequence
# Since for each testset example may yield more than one example, because of the different
# pairings created from several input segments (Q+A1, Q+A2, ...), it may be helpful to
# Create a lookup dictionary that maps the supplied sequence IDs to the original example IDs
# e.g. CSQA creates 5 sequences from each example. As such, content from example 1 is
# ID 0, 1, 2, 3, 4 in the supplied examples to OverlapyTestSet
print(f"PIQA Matches: {list(piqa_testset.get_matches(matches))}")
