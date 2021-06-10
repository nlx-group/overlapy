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
    return [
        word.lower() for word in
        nltk.word_tokenize(s)
        if word.isalnum()
    ]


# The next three functions transform a dataset example into sequences of tokens.
# This example performs a data contamination study on the 3 following tasks:
#      * AI2 Reasoning Challenge (ARC) - Clark, Peter, et al. "Think you have solved question answering? try arc, the ai2 reasoning challenge." arXiv preprint arXiv:1803.05457 (2018).
#      * CommonsenseQA (CSQA) - Talmor, Alon, et al. "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.
#      * Physical Interaction: Question Answering (PIQA) - Bisk, Yonatan, et al. "Piqa: Reasoning about physical commonsense in natural language." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 05. 2020.

# Datasets can have several input segments. As an example, CommonsenseQA has 1 Question and 5 candidate answers.
# As such we tokenize the 5 possible pairings, as we can have 5 possible overlaps:
#    1. Q + A1
#    2. Q + A2
#    3. Q + A3
#    4. Q + A4
#    5. Q + A5

# The same rationale is applied to every dataset.


def arc_example_to_tokens(example):
    tokens = []
    for i in range(len(example["choices"]["label"])):
        tokens.append(tokenizer(example["question"]) + tokenizer(example["choices"]["text"][i]))
    return tokens


def csqa_example_to_tokens(example):
    tokens = []
    for i in range(len(example["choices"]["label"])):
        tokens.append(tokenizer(example["question"]) + tokenizer(example["choices"]["text"][i]))
    return tokens


def piqa_example_to_tokens(example):
    tokens = []
    for i in range(1, 3):
        tokens.append(tokenizer(example["goal"]) + tokenizer(example[f"sol{i}"]))
    return tokens


# We load the datasets, turn them into sequences of tokens using our example_to_tokens functions
# And then create an OverlapyTestSet object

arc = load_dataset("ai2_arc", "ARC-Challenge")
examples = list(chain(*[arc_example_to_tokens(example) for example in arc["test"]]))
arc_testset = OverlapyTestSet("arc", examples=examples)
print("ARC Testset")
print(f"N value: {arc_testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(arc_testset.ngrams()))))}")


csqa = load_dataset("commonsense_qa")
examples = list(chain(*[csqa_example_to_tokens(example) for example in csqa["validation"]]))
csqa_testset = OverlapyTestSet("csqa", examples=examples)
print("CSQA Testset")
print(f"N value: {csqa_testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(csqa_testset.ngrams()))))}")

piqa = load_dataset("piqa")
examples = list(chain(*[piqa_example_to_tokens(example) for example in piqa["validation"]]))
piqa_testset = OverlapyTestSet("piqa", examples=examples)
print("PIQA Testset")
print(f"N value: {piqa_testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(piqa_testset.ngrams()))))}")


# We're defining a wrapper for a HuggingFace dataset to make it compatible with overlapy.
# Overlapy expects __getitem__ to return the example's text.
# However, HuggingFace Dataset's __getitem__ returns a dictionary, and the text is
# Stored in the key "text". As such, we define a wrapper that receives a HF Dataset
# And the __getitem__ receives an idx, accesses the dataset idx and selects the "text" key value

class HuggingFaceDatasetWrapper:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds[idx]["text"]

    def __len__(self):
        return len(self.ds)


# We are analyzing overlaps with OpenWebText (https://skylion007.github.io/OpenWebTextCorpus/)
# This language modeling dataset was used in models such as the GPT series and RoBERTa.
dataset = load_dataset("openwebtext")["train"]

# We create an Overlapy object, handing four arguments:
#   * Testsets: A list of OverlapyTestSet objects that we want to study.
#   * Dataset: Dataset we want to calculate collisions with
#   * Tokenizer: Tokenization function
#   * n_workers: Number of worker processes to use
# It's advisable to stay below 32 workers for HuggingFace datasets.
# In our experience, more than that leads to race conditions and
# it permanently stops.
overlapy = Overlapy(
    testsets=[arc_testset, csqa_testset, piqa_testset],
    dataset=HuggingFaceDatasetWrapper(dataset),
    tokenizer=tokenizer,
    n_workers=32,
)
# Let's run and get the matches
matches = overlapy.run()

# Using the matches dictionary, we use the OverlapyTestSet object from each testset
# To obtain information about which examples were found to have matches with
# OpenWebText

# The output follows the structure: Example ID, Ngram, Match position within example sequence
# Since for each testset example may yield more than one example, because of the different
# pairings created from several input segments (Q+A1, Q+A2, ...), it may be helpful to
# Create a lookup dictionary that maps the supplied sequence IDs to the original example IDs
# e.g. CSQA creates 5 sequences from each example. As such, content from example 1 is
# ID 0, 1, 2, 3, 4 in the supplied examples to OverlapyTestSet
print(f"ARC Matches: {list(arc_testset.get_matches(matches))}")
print(f"PIQA Matches: {list(piqa_testset.get_matches(matches))}")
print(f"CSQA Matches: {list(csqa_testset.get_matches(matches))}")
