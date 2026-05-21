import unittest
from unittest.mock import patch, MagicMock
import os
import sys

class TestFuzzyMatchColumns(unittest.TestCase):

    def setUp(self):
        # We use patch.dict for os.environ within the test so it restores automatically
        pass

    def test_fuzzy_match_columns_logic(self):
        mock_streamlit = MagicMock()

        def mock_cache_decorator(func=None, **kwargs):
            if func is None:
                def wrapper(f):
                    f.clear = MagicMock()
                    return f
                return wrapper
            func.clear = MagicMock()
            return func

        class MockCacheData:
            def __call__(self, func=None, **kwargs):
                return mock_cache_decorator(func, **kwargs)
            def clear(self):
                pass

        class MockCacheResource:
            def __call__(self, func=None, **kwargs):
                return mock_cache_decorator(func, **kwargs)
            def clear(self):
                pass

        mock_streamlit.cache_data = MockCacheData()
        mock_streamlit.cache_resource = MockCacheResource()

        def mock_columns(spec):
            if isinstance(spec, int):
                return [MagicMock() for _ in range(spec)]
            elif isinstance(spec, (list, tuple)):
                return [MagicMock() for _ in spec]
            return [MagicMock()]

        mock_streamlit.columns = mock_columns
        mock_streamlit.session_state = {}

        class FakeFuzz:
            token_sort_ratio = "token_sort_ratio"

        class FakeProcess:
            @staticmethod
            def extractOne(query, choices, scorer=None):
                results = {
                    "high_match": ("k_high", 95),
                    "low_match": ("k_low", 75),
                    "exact_match": ("k_exact", 100),
                    "no_match": None
                }
                return results.get(query)

        mock_thefuzz = MagicMock()
        mock_thefuzz.fuzz = FakeFuzz
        mock_thefuzz.process = FakeProcess

        modules_to_patch = {
            'streamlit': mock_streamlit,
            'pandas': MagicMock(),
            'numpy': MagicMock(),
            'scipy': MagicMock(),
            'scipy.stats': MagicMock(),
            'requests': MagicMock(),
            'zipfile': MagicMock(),
            'shutil': MagicMock(),
            'plotly': MagicMock(),
            'plotly.express': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'PyPDF2': MagicMock(),
            'docx': MagicMock(),
            'thefuzz': mock_thefuzz,
            'thefuzz.fuzz': FakeFuzz,
            'thefuzz.process': FakeProcess,
        }

        # Also patch environment variables to prevent Kaggle API errors
        env_vars = {'KAGGLE_USERNAME': 'dummy_user', 'KAGGLE_KEY': 'dummy_key'}

        with patch.dict('sys.modules', modules_to_patch), patch.dict(os.environ, env_vars):
            import app

            user_cols = ["high_match", "low_match", "exact_match", "no_match", "unknown"]
            kaggle_cols = ["k_high", "k_low", "k_exact", "other"]

            mapping = app.fuzzy_match_columns(user_cols, kaggle_cols)

            self.assertIn("high_match", mapping)
            self.assertEqual(mapping["high_match"], "k_high")

            self.assertIn("exact_match", mapping)
            self.assertEqual(mapping["exact_match"], "k_exact")

            self.assertNotIn("low_match", mapping)
            self.assertNotIn("no_match", mapping)
            self.assertNotIn("unknown", mapping)

            self.assertEqual(len(mapping), 2)

            mapping_custom = app.fuzzy_match_columns(user_cols, kaggle_cols, threshold=70)
            self.assertIn("low_match", mapping_custom)
            self.assertEqual(len(mapping_custom), 3)

if __name__ == '__main__':
    unittest.main()
