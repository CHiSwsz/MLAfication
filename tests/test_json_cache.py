"""Regression checks for replacing pickle-backed evaluation response caches."""
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sqlite3
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'MHA2MLA-VLM/lmms-eval/lmms_eval/caching/json_cache.py'
spec = importlib.util.spec_from_file_location('response_json_cache', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class JsonCacheTests(unittest.TestCase):
    def test_response_types_persist_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'responses.db')
            cache = module.JsonCache(filename)
            values = {'generation': '中文 response', 'likelihood': (-1.5, True), 'nested': {'answer': ['a', (2, False)]}, 'infinite': (-math.inf, False)}
            for key, value in values.items():
                cache[key] = value
            cache.commit()
            cache.close()
            reopened = module.JsonCache(filename)
            for key, value in values.items():
                self.assertEqual(reopened[key], value)
            self.assertIsInstance(reopened['likelihood'], tuple)
            self.assertEqual(len(reopened), len(values))
            del reopened['generation']
            self.assertNotIn('generation', reopened)
            reopened.close()

    def test_legacy_pickle_table_is_preserved_and_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'responses.db')
            old = pickle.dumps({'old': 'response'})
            with sqlite3.connect(filename) as database:
                database.execute('CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)')
                database.execute('INSERT INTO unnamed VALUES (?, ?)', ('legacy', old))
            cache = module.JsonCache(filename)
            self.assertNotIn('legacy', cache)
            cache['new'] = 'JSON only'
            cache.close()
            with sqlite3.connect(filename) as database:
                self.assertEqual(database.execute('SELECT value FROM unnamed').fetchone()[0], old)

    def test_non_json_cache_data_is_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'responses.db')
            cache = module.JsonCache(filename)
            with sqlite3.connect(filename) as database:
                database.execute('INSERT INTO responses_json_v1 VALUES (?, ?)', ('invalid', pickle.dumps(('data',))))
            with self.assertRaises((ValueError, UnicodeError)):
                cache['invalid']
            with self.assertRaises(TypeError):
                cache['object'] = object()
            cache.close()


if __name__ == '__main__':
    unittest.main()
