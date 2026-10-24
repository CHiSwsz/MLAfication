#!/usr/bin/env python3
"""Exercise NLTK evaluation primitives and the patched resource-path boundary."""
from pathlib import Path
import tempfile

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.translate.bleu_score import sentence_bleu
import nltk.pathsec

assert tuple(int(part) for part in nltk.__version__.split('.')[:3]) >= (3, 10, 4)
assert nltk.pathsec.ENFORCE is True
assert TreebankWordTokenizer().tokenize('MLA keeps the KV cache small.') == [
    'MLA', 'keeps', 'the', 'KV', 'cache', 'small', '.'
]
assert PorterStemmer().stem('running') == 'run'
assert sentence_bleu([['a', 'b', 'c', 'd']], ['a', 'b', 'c', 'd']) == 1.0
with tempfile.TemporaryDirectory() as directory:
    outside = Path(directory) / 'outside-nltk-data.txt'
    outside.write_text('Security regression test, no private data.')
    try:
        nltk.data.load(str(outside), format='text')
    except ValueError as exc:
        assert 'Unsafe resource path' in str(exc) or 'Security Violation' in str(exc)
    else:
        raise AssertionError('NLTK read a file outside its trusted resource roots')
print(f'NLTK {nltk.__version__}: evaluation smoke checks and resource-path guard passed')
