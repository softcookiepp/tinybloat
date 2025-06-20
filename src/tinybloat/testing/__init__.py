import pytest

def run_tests():
	pytest.main(["--pyargs", "tinybloat.testing.example_tests"])
