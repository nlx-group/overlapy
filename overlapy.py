import collections
from itertools import chain
from multiprocessing import Pool, cpu_count
from typing import Iterable

from stringology.ac import AhoCorasick
from stringology.ngrams import all_ngrams

try:
    from tqdm.auto import tqdm

    try:
        # HOTFIX: in case the environment is jupyterlab
        # this is required to have multiple tqdm progress bars
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            tqdm_original = tqdm

            def tqdm(*args, **kwargs):
                print(" ", end="", flush=True)
                return tqdm_original(*args, **kwargs)

    except NameError:
        pass

except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


__version__ = "0.0.1"
__author__ = "Ruben Branco, Luís Gomes"
__copyright__ = "copyright © 2021, Ruben Branco, Luís Gomes, all rights reserved"


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

    def add_example(self, example):
        self.examples.append(example)

    @staticmethod
    def get_percentile(values, percentile):
        values.sort()
        i = int(len(values) * percentile / 100)
        return values[min(i, len(values) - 1)]

    def compute_n(self):
        """
        Compute the optimal size of N-Grams for data contamination studies, for this testset.

        Based on the methodology set forth by:
        Brown, Tom B., et al. "Language models are few-shot learners." arXiv preprint arXiv:2005.14165 (2020).
        https://arxiv.org/abs/2005.14165
        """
        hist = sorted(map(len, self.examples))
        n = OverlapyTestSet.get_percentile(hist, self.percentile)
        return min(max(self.min_n, n), self.max_n)

    def ngrams(self):
        """
        Compute ngrams of size N (see compute_n()) for each example.
        """
        n = self.compute_n()
        for example in self.examples:
            yield from all_ngrams(example, minn=n, maxn=n)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    def get_matches(self, matches):
        """
        Given a dictionary of matches, retrieve the matched examples, together with
        the matched ngram and position in the sequence.

        The structure of the output is the following:
        Example ID, Ngram, Match position within example sequence.
        """
        ac = AhoCorasick(matches.keys())

        for i, example in enumerate(self.examples):
            for ngram, position in ac(example):
                yield i, ngram, position


class OverlapyNgramMatcher:
    def __init__(self, ngrams: set):
        self.ac = AhoCorasick(ngrams)

    def __call__(self, examples):
        matches = collections.defaultdict(list)
        for i, example in enumerate(examples):
            for ngram, _ in self.ac(example):
                matches[ngram].append(i)
        return matches


class Overlapy:
    def __init__(
        self, testsets: list, dataset: Iterable, n_workers=cpu_count()
    ):
        assert n_workers <= cpu_count()
        self.dataset = dataset
        self.testsets = testsets
        self.testset_ngrams = set(
            map(tuple, chain(*[list(testset.ngrams()) for testset in testsets]))
        )
        self.n_workers = n_workers

    def _calculate_chunk_matches(self, args):
        """
        Calculates matches between the testset ngrams and a chunk of the given dataset.

        This function is executed by each worker from a pool of workers (processes).
        """
        matches = collections.defaultdict(list)
        idxs, n_worker = args
        matcher = OverlapyNgramMatcher(self.testset_ngrams)

        for idx in tqdm(
            idxs, total=len(idxs), position=n_worker + 1, desc=f"Worker #{n_worker}"
        ):
            matched = matcher([self.dataset[idx]])
            for ngram, positions in matched.items():
                matches[ngram].extend([idx] * len(positions))
        return matches

    def run(self):
        """
        Calculates ngram matches between the supplied dataset and the testsets, in a parallel manner.

        A pool of workers is leveraged to calculate matches between different chunks of the dataset.
        """
        pool = Pool(self.n_workers)
        matches = collections.defaultdict(list)

        for d in tqdm(
            pool.imap_unordered(
                self._calculate_chunk_matches,
                zip(
                    list_split(list(range(len(self.dataset))), self.n_workers),
                    list(range(self.n_workers)),
                ),
            ),
            total=self.n_workers,
            position=0,
            desc="Global progress",
        ):
            for ngram, positions in d.items():
                matches[ngram].extend(positions)

        pool.close()
        pool.join()

        return matches


def list_split(lst, sections):
    """
    Splits a list into N sections. From https://stackoverflow.com/a/2135920.

    Examples:

    >>> list_split(list(range(10)), 3)
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    >>> list_split(list(range(20)), 4)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    """
    k, m = divmod(len(lst), sections)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(sections)
    ]
