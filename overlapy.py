import collections

from stringology.ac import AhoCorasick
from stringology.ngrams import all_ngrams

__version__ = "0.0.1"
__author__ = "Ruben Branco, Luís Gomes"
__copyright__ = "copyright © 2021, Ruben Branco, Luís Gomes, all rights reserved"


def get_percentile(values, percentile):
    values.sort()
    i = int(len(values) * percentile / 100)
    return values[min(i, len(values) - 1)]


class OverlapyTestSet:
    def __init__(self, name, min_n=8, max_n=13, percentile=5, examples=None):
        assert isinstance(min_n, int) and isinstance(max_n, int)
        assert 1 <= min_n <= max_n
        assert 0 <= percentile <= 100
        self.name = name
        self.min_n = min_n
        self.max_n = max_n
        self.percentile = percentile
        self.examples = examples or []

    def add_example(self, *seqs):
        self.examples.append(seqs)

    def seqs(self):
        for example in self.examples:
            yield from example

    def compute_n(self):
        hist = sorted(map(len, self.seqs()))
        return min(max(self.min_n, get_percentile(hist, self.percentile)), self.max_n)

    def ngrams(self):
        n = self.compute_n()
        for seq in self.seqs():
            yield from all_ngrams(seq, minn=n, maxn=n)


class OverlapyNgramMatcher:
    def __init__(self, ngrams: set):
        self.ac = AhoCorasick(ngrams)

    def __call__(self, examples):
        matches = collections.defaultdict(list)
        for i, example in enumerate(examples):
            for ngram, _ in self.ac(example):
                matches[ngram].append(i)
        return matches
