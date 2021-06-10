from os.path import join

from overlapy import OverlapyTestSet, Overlapy

# This example uses the same pretraining dataset and testset
# as synthetic_text_example.py, but stored in files for demonstration purposes
# As such, 3 examples from the testset are expected to be matched

def tokenizer(s):
    # Simple tokenization by whitespace.
    return s.split()


with open(join("files", "testset.txt")) as fr:
    testset_examples = list(map(str.rstrip, fr.readlines()))

# We'll override the parameter min_n and set it to 1 as we want the ngram value to be allowed
# to be less than 8. The testset examples were constructed for it to be 4, actually.
testset = OverlapyTestSet("test", min_n=1, examples=[tokenizer(s) for s in testset_examples])
print(f"N value: {testset.compute_n()}")
print(f"# NGrams: {len(set(map(tuple, list(testset.ngrams()))))}")


class SyntheticPreTrainingDataset:
    def __getitem__(self, idx):
        # Our pretraining dataset is split into 5 files.
        # For large datasets, this is likely to happen.
        # Here we offer a simple example where each file has just one sentence
        # But the directory structure and file structure can be arbitrarily complex
        with open(join("files", f"pretraining_dataset.{idx}.txt")) as fr:
            return fr.read().rstrip()

    def __len__(self):
        # We could list the directory
        # But we know its 5 files so why not just explicitly say
        return 5

# We create an Overlapy object, handing four arguments:
#   * Testsets: A list of OverlapyTestSet objects that we want to study.
#   * Dataset: Dataset we want to calculate collisions with
#   * Tokenizer: Tokenization function
#   * n_workers: Number of worker processes to use
overlapy = Overlapy(
    testsets=[testset],
    dataset=SyntheticPreTrainingDataset(),
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
