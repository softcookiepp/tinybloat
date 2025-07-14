"""Main"""

# imports: library
from argparse import ArgumentParser
from .testing import run_tests

from . import version
from . import quantization


def main() -> None:
	"""Main"""
	parser = ArgumentParser(prog=version.PROGRAM_NAME)

	parser.add_argument('--version',
						help='Display version',
						action='store_true',
						dest='version')
	parser.add_argument("--test", help="Run tests", action = "store_true", dest = "test")

	args = parser.parse_args()

	if args.version:
		print(f'{version.PROGRAM_NAME} {version.__version__}')
		return
	if args.test:
		run_tests()

if __name__ == '__main__':
	main()
