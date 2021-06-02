import pytest
from overlapy import OverlapyNgramMatcher, OverlapyTestSet

@pytest.fixture
def examples1():
    return [
        "12345",
        "1234",
        "123",
        "12",
        "1",
    ]


@pytest.fixture
def ts1(examples1):
    ts = OverlapyTestSet("ts1", min_n=1, max_n=1000, percentile=50)
    for example in examples1:
        ts.add_example(example)
    return ts


def test_compute_n(ts1):
    ts1.percentile = 0
    assert 1 == ts1.compute_n()
    ts1.percentile = 10
    assert 1 == ts1.compute_n()
    ts1.percentile = 19
    assert 1 == ts1.compute_n()
    ts1.percentile = 20
    assert 2 == ts1.compute_n()
    ts1.percentile = 39
    assert 2 == ts1.compute_n()
    ts1.percentile = 40
    assert 3 == ts1.compute_n()
    ts1.percentile = 59
    assert 3 == ts1.compute_n()
    ts1.percentile = 60
    assert 4 == ts1.compute_n()
    ts1.percentile = 79
    assert 4 == ts1.compute_n()
    ts1.percentile = 80
    assert 5 == ts1.compute_n()
    ts1.percentile = 99
    assert 5 == ts1.compute_n()
    ts1.percentile = 100
    assert 5 == ts1.compute_n()


def test_unigrams(examples1, ts1):
    unigrams = [
        unigram
        for example in examples1
        for unigram in example
    ]
    ts1.min_n=1
    ts1.max_n=1
    assert sorted(unigrams) == sorted(ts1.ngrams())


def test_bigrams(examples1, ts1):
    bigrams = [
        bigram
        for example in examples1
        for bigram in map(''.join, zip(example[:-1], example[1:]))
    ]
    ts1.min_n=2
    ts1.max_n=2
    assert sorted(bigrams) == sorted(ts1.ngrams())


def test_match(examples1, ts1):
    train_examples = [
        example + example for example in examples1
    ]
    match = OverlapyNgramMatcher(ts1.ngrams())
    matched = match(train_examples)
    assert matched == {'123': [0, 0, 1, 1, 2, 2], '234': [0, 0, 1, 1], '345': [0, 0]}
