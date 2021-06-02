import collections

from stringology.ac import AhoCorasick
from stringology.ngrams import all_ngrams


class OverlapyTestSet:
    def __init__(self, name, examples=None):
        self.name = name
        self.examples = examples or []

    def add_example(self, *seqs):
        self.examples.append(seqs)

    def compute_n(self, min_n=8, max_n=13, percentile=5):
        seq_lens = [len(seq) for example in self.examples for seq in example]
        seq_lens.sort()
        pos = int(len(seq_lens) * percentile / 100)
        return min(max(min_n, seq_lens[pos]), max_n)

    def ngrams(self, n):
        for example in self.examples:
            for seq in example:
                yield from all_ngrams(seq, minn=n, maxn=n)


class OverlapyTrainSetLoader:
    def __init__(self, name):
        self.name = name

    def examples(self):
        raise NotImplemented


class OverlapyNgramIndex:
    def __init__(self, min_n=8, max_n=13):
        self.min_n = min_n
        self.max_n = max_n
        self.ac = None
        self.ngrams = set()

    def add_testset(self, testset, min_n=None, max_n=None):
        assert self.ngrams is not None
        n = testset.compute_n(
            self.min_n if min_n is None else min_n,
            self.max_n if max_n is None else max_n,
        )
        self.ngrams.update(testset.ngrams(n))

    def build_index(self):
        self.ac = AhoCorasick(self.ngrams)
        self.ngrams = None

    def match(self, trainset):
        matches = collections.defaultdict(list)
        for i, example in enumerate(trainset.examples()):
            for ngram, position in self.ac(example):
                matches[ngram].append(i)
        return matches
