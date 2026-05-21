import unittest
from unittest.mock import MagicMock, patch
import os

class TestFuzzyMatchColumns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set dummy env vars to prevent Kaggle errors
        os.environ['KAGGLE_USERNAME'] = 'dummy'
        os.environ['KAGGLE_KEY'] = 'dummy'

    def setUp(self):
        # Patch sys.modules for missing dependencies

        mock_streamlit = MagicMock()
        def mock_columns(n):
            if isinstance(n, int):
                return [MagicMock() for _ in range(n)]
            return [MagicMock() for _ in n]

        mock_streamlit.columns = mock_columns

        # Mocks for cache decorators returning the original function
        def mock_cache_decorator(*args, **kwargs):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator

        mock_streamlit.cache_data = mock_cache_decorator
        mock_streamlit.cache_resource = mock_cache_decorator

        # Explicit mock to prevent clear errors on cache_data
        mock_cache_data_func = MagicMock()
        mock_cache_data_func.clear = MagicMock()
        mock_streamlit.cache_data = mock_cache_decorator
        mock_streamlit.cache_data.clear = mock_cache_data_func.clear

        self.mock_modules = {
            'streamlit': mock_streamlit,
            'pandas': MagicMock(),
            'numpy': MagicMock(),
            'scipy': MagicMock(),
            'scipy.stats': MagicMock(),
            'requests': MagicMock(),
            'thefuzz': MagicMock(),
            'thefuzz.fuzz': MagicMock(),
            'thefuzz.process': MagicMock(),
            'zipfile': MagicMock(),
            'shutil': MagicMock(),
            'plotly': MagicMock(),
            'plotly.express': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'PyPDF2': MagicMock(),
            'docx': MagicMock()
        }
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

        import app
        self.app = app

    def tearDown(self):
        self.patcher.stop()

    def test_fuzzy_match_columns_happy_path(self):
        """Test that matches above the threshold are correctly mapped."""
        def fake_extractOne(query, choices, scorer=None):
            matches = {
                "Speed": ("Velocity", 90),
                "Temp": ("Temperature", 85),
            }
            return matches.get(query, ("Unknown", 0))

        self.app.process.extractOne = fake_extractOne

        user_cols = ["Speed", "Temp", "UnknownCol"]
        kaggle_cols = ["Velocity", "Temperature", "Pressure"]

        result = self.app.fuzzy_match_columns(user_cols, kaggle_cols, threshold=80)
        self.assertEqual(result, {"Speed": "Velocity", "Temp": "Temperature"})

    def test_fuzzy_match_columns_below_threshold(self):
        """Test that matches below the threshold are ignored."""
        def fake_extractOne(query, choices, scorer=None):
            return ("SomeColumn", 79)

        self.app.process.extractOne = fake_extractOne
        result = self.app.fuzzy_match_columns(["UserCol"], ["SomeColumn"])
        self.assertEqual(result, {})

    def test_fuzzy_match_columns_none_match(self):
        """Test handling of None returns from extractOne."""
        def fake_extractOne(query, choices, scorer=None):
            return None

        self.app.process.extractOne = fake_extractOne
        result = self.app.fuzzy_match_columns(["UserCol"], [])
        self.assertEqual(result, {})

    def test_fuzzy_match_columns_custom_threshold(self):
        """Test that custom thresholds are respected."""
        def fake_extractOne(query, choices, scorer=None):
            return ("KaggleCol", 65)

        self.app.process.extractOne = fake_extractOne

        # Should match
        result = self.app.fuzzy_match_columns(["UserCol"], ["KaggleCol"], threshold=60)
        self.assertEqual(result, {"UserCol": "KaggleCol"})

        # Should not match
        result = self.app.fuzzy_match_columns(["UserCol"], ["KaggleCol"], threshold=70)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
