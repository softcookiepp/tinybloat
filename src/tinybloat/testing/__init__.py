import pytest

def run_tests():
	pytest.main(["--pyargs", "tinybloat.testing.operator_tests", "tinybloat.testing.module_tests"])
