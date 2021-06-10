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

# We create an Overlapy object, handing four arguments:
#   * Testsets: A list of OverlapyTestSet objects that we want to study.
#   * Dataset: Dataset we want to calculate collisions with
#   * Tokenizer: Tokenization function
#   * n_workers: Number of worker processes to use
overlapy = Overlapy(
    testsets=[testset],
    dataset=pretraining_dataset,
    tokenizer=tokenizer,
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
