<p align="center"><img src="logo-w-text.png" alt="Overlapy Logo" /></p>

--------------------------------------------------------------------------------

<p align="center">
  <a href="#about">About</a> ⚭
  <a href="#installation">Installation</a> ⚭
  <a href="#usage">Usage</a> ⚭
  <a href="#citation">Citation</a>
</p>

## About

Overlapy is a Python package developed to evaluate textual overlap (N-Grams) between two volumes of text. In fact, it comes from the necessity of evaluating "data contamination" between pre-training datasets for Language Models and testsets of NLP tasks. This problem is starting to become relevant: as models become ever larger, rapidly entering the trillions of parameters mark, they can fit larger pre-training language modelling datasets, which have started to inch closer to the terabytes mark.

The web is a source of nearly unlimited natural language text, making it one of the favourite sources to obtain unlabelled text. Websites like Reddit (<https://reddit.com/>) aggregate content and outbound links in inconcievable amounts. However, these resources are not exclusive to the language modelling task, and other tasks use them to construct even labelled datasets. As web crawlers extend their scrapped nodes, the probability of obtaining text that has been used in other tasks grows larger. With the capability of these models to memorize spans of text, it can just so happen that specific spans from examples of a tasks' testset could have been found in the pre-training dataset. The language model could have memorized it, making it previously seen data less than ideal as we want to test our models with unseen (o.o.d) data. This constitutes a problem for the present and future.

The methodology followed for this implementation is described in GPT-3's paper appendix (<https://arxiv.org/abs/2005.14165>). It can be decomposed into three main parts: tokenize, choosing N-Gram size, calculate N-Gram collisions between pre-training datasets and testsets.

1. A token is considered an alphanumeric character, delimited by whitespace, and lowercased. In overlapy, the tokenization function is arbitrary (user-defined), and does not need to follow this definition.
2. N-Gram size is determined to be the 5th percentile of the distribution of testset examples lengths. The authors set a minimum size of 8 and maximum size of 13. We follow this definition, however, allow the user to redefine the percentile, minimum and maximum size.
3. Collisions are calculated by our package using the Aho-Corasick algorithm (<https://dl.acm.org/doi/10.1145/360825.360855>). The testsets are decomposed into N-Grams. Subsequently, we distribute the pre-training dataset to a pool of workers, calculating matches between the testset N-Grams and examples from the pre-training dataset.


## Installation

Packaged developed to work with Python 3+. Some examples require Python 3.6+ and nltk (<http://www.nltk.org/>) installed.

tqdm (<https://github.com/tqdm/tqdm>) not mandatory to have installed but is recommended to track the progress, especially for jobs with several hundreds of gigabytes of text.

```bash
pip install overlapy
```

## Usage

It follows the contents of an usage example from one of our examples found [here](examples/).

```python
from overlapy import OverlapyTestSet, Overlapy

pretraining_dataset = [
    "A B A C D E F G",
    "A C F J K H E",
    "V L N M Q",
    "A B A C Ç T Z V E",
    "L M N O P",
]

testset_examples = [
    "B A B A C O Q W R",  # Match A B A C with #1, #4 from pretraining_dataset
    "O P Q F J K H",  # Match F J K H with #2 from pretraining_dataset
    "W E R E",  # No match
    "I E T Z V E L",  # Match T Z V E with #4 from pretraining_dataset
    "K E K W",  # No match
]
# Total examples matched: 3


def tokenizer(s):
    # Simple tokenization by whitespace.
    return s.split()


# We'll override the parameter min_n and set it to 1 as we want the ngram value to be allowed
# to be less than 8. The testset examples were constructed for it to be 4, actually.
testset = OverlapyTestSet(
    "test", min_n=1, examples=[tokenizer(s) for s in testset_examples]
)
print(f"N value: {testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(testset.ngrams()))))}")

# We create an Overlapy object, handing three arguments:
#   * Testsets: A list of OverlapyTestSet objects that we want to study.
#   * Dataset: Dataset we want to calculate collisions with
#   * n_workers: Number of worker processes to use
overlapy = Overlapy(
    testsets=[testset],
    dataset=[tokenizer(s) for s in pretraining_dataset],
    n_workers=2,
)
# Let's run and get the matches
matches = overlapy.run()

# We should be getting 3 testset examples that have been flagged for matches.
#    #0 matches on A B A C
#    #1 matches on F J K H
#    #3 matches on T V Z E
# As we had noted above
print(f"Matches: {list(testset.get_matches(matches))}")
```

## Citation

Bibtex citation will be available soon.
