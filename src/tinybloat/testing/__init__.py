"""
Unit tests for all the functionality implemented in tinybloat.
Contained in the same module so that any ordinary user may run the tests.
"""
import pytest

_gadget_available = False
try:
	import gadget
	_gadget_available = False
except ImportError:
	print("gadget-ml is required for testing quantization, but was not found.\nCurrently the official version is broken, so there is nothing that can be done for now :c ")

def run_tests():
	tests = ["--pyargs", "tinybloat.testing.operator_tests", "tinybloat.testing.module_tests"]
	if _gadget_available:
		tests.append("tinybloat.testing.quantization_tests")
	pytest.main(tests)
