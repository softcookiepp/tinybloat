# tinybloat
A library to extend tinygrad functionality with stuff that is probably too bloated for tinygrad.

## Installation
Just run `pip install -U git+https://github.com/softcookiepp/tinybloat.git`

## Documentation
Simply open docs/index.html in your browser of choice, or run the following command to open a documentation webserver:
`pdoc src/tinybloat`

## Contributing
Contributions to this codebase require only 2 things:
1) New functionality must have docstrings that describe its use in some capacity.
2) Unit tests must be written to validate it against the original implementation (usually pytorch or numpy.)

## Roadmap
These are functionality that should be reimplemented in the future:
### High priority:
- Function for reliably determining the supported data types of a given hardware device
- Conversion of fp8e4m3 and fp8e5m2 tensors to float32 even if no available devices support either fp8 data type
- Finish implementing ComplexTensor operators such that it reaches feature parity with tinygrad.Tensor
### Lower priority:
- [All the loss functions from pytorch](https://docs.pytorch.org/docs/stable/nn.html#loss-functions)
- [torch.nn.CosineSimilarity](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity)
- [torch.nn.PairwiseDistance](https://docs.pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html#torch.nn.PairwiseDistance)
- any other functionality that torch or other ML libraries have
