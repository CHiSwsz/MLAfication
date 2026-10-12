# Third-Party Notices

MLAfication incorporates or adapts portions of the following projects. These
portions remain subject to their original copyright and license terms.

## Hugging Face LightEval

Portions of `mha_gqa/eval/eval_v2.py` and `qknorm/eval/eval_v2.py` are adapted
from LightEval v0.6.0:

https://github.com/huggingface/lighteval/tree/v0.6.0

MIT License

Copyright (c) 2024 Hugging Face

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Hugging Face Nanotron

The learning-rate scheduler implementation in
`mha_gqa/src/mha2mla/helpers.py` and `qknorm/src/mha2mla/helpers.py` is adapted
from Nanotron v0.5:

https://github.com/huggingface/nanotron/tree/v0.5

Nanotron is licensed under the Apache License, Version 2.0. The complete
Apache-2.0 terms are included in this repository's `LICENSE` file. The adapted
implementation was modified to operate on standard optimizer parameter groups.

## Hugging Face Datasets

`qknorm/eval/LongBenchData/LongBench.py` retains its upstream copyright and
Apache-2.0 license header. The complete Apache-2.0 terms are included in this
repository's `LICENSE` file.
