import unittest
from unittest.mock import patch, MagicMock
import sys
import os

class TestSemanticMatchColumnsWithLLM(unittest.TestCase):
    def setUp(self):
        # We need to mock all these before importing app.py
        mock_st = MagicMock()
        mock_cache_data = MagicMock()
        def mock_cache_decorator(*args, **kwargs):
            def decorator(func):
                func.clear = MagicMock()
                return func
            if len(args) == 1 and callable(args[0]):
                args[0].clear = MagicMock()
                return args[0]
            return decorator

        mock_cache_data.side_effect = mock_cache_decorator
        mock_st.cache_data = mock_cache_data

        def mock_columns(spec):
            if isinstance(spec, int):
                return [MagicMock() for _ in range(spec)]
            else:
                return [MagicMock() for _ in spec]

        mock_st.columns = mock_columns

        sys.modules['streamlit'] = mock_st
        sys.modules['pandas'] = MagicMock()
        sys.modules['numpy'] = MagicMock()
        sys.modules['scipy'] = MagicMock()
        sys.modules['scipy.stats'] = MagicMock()
        sys.modules['requests'] = MagicMock()
        sys.modules['thefuzz'] = MagicMock()
        sys.modules['plotly'] = MagicMock()
        sys.modules['plotly.express'] = MagicMock()
        sys.modules['plotly.graph_objects'] = MagicMock()
        sys.modules['PyPDF2'] = MagicMock()
        sys.modules['docx'] = MagicMock()
        sys.modules['kaggle'] = MagicMock()

        os.environ['KAGGLE_USERNAME'] = 'dummy'
        os.environ['KAGGLE_KEY'] = 'dummy'

        # Now import app
        import app
        self.app = app

    def test_semantic_match_columns_with_llm_exception(self):
        # Mock call_llm to raise Exception
        with patch.object(self.app, 'call_llm', side_effect=Exception("API Error")):
            result = self.app.semantic_match_columns_with_llm(
                user_cols=['A'], kaggle_cols=['B'],
                llm_provider='test', llm_url='test', model_name='test'
            )
            self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
