from overlapy import OverlapyTestSet


def test_compute_n():
    ts = OverlapyTestSet("T1")
    ts.add_example("12345")
    ts.add_example("1234")
    ts.add_example("123")
    ts.add_example("12")
    ts.add_example("1")
    assert 3 == ts.compute_n(percentile=50, min_n=0, max_n=1000)
