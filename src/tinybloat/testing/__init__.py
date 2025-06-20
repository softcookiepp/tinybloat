"""
Unit tests for all the functionality implemented in tinybloat.
Contained in the same module so that any ordinary user may run the tests.
"""
import pytest

def run_tests():
	pytest.main(["--pyargs", "tinybloat.testing.operator_tests", "tinybloat.testing.module_tests"])
