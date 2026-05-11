import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.dont_write_bytecode = True
os.environ['KAGGLE_USERNAME'] = 'mock'
os.environ['KAGGLE_KEY'] = 'mock'

# Mock streamlit and other dependencies
mock_st = MagicMock()

class MockCache:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def clear(self):
        pass

def mock_cache_data(func):
    return MockCache(func)
mock_cache_data.clear = MagicMock()

def mock_cache_resource(func):
    return MockCache(func)
mock_cache_resource.clear = MagicMock()

mock_st.cache_data = mock_cache_data
mock_st.cache_resource = mock_cache_resource

def mock_columns(spec):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    elif isinstance(spec, list):
        return [MagicMock() for _ in range(len(spec))]
    return [MagicMock(), MagicMock()]
mock_st.columns = mock_columns

sys.modules['streamlit'] = mock_st
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['thefuzz'] = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()
sys.modules['docx'] = MagicMock()
sys.modules['kaggle'] = MagicMock()
sys.modules['kaggle.api.kaggle_api_extended'] = MagicMock()

# Now we can import app
import app

class TestSemanticMatchColumnsWithLlm(unittest.TestCase):
    @patch('app.call_llm')
    def test_semantic_match_columns_with_llm_exception(self, mock_call_llm):
        """Test fallback behavior when call_llm raises an exception."""
        mock_call_llm.side_effect = Exception("API Error")
        result = app.semantic_match_columns_with_llm(
            ['col1'], ['col2'], 'OpenAI', 'url', 'model'
        )
        self.assertEqual(result, {})

    @patch('app.call_llm')
    def test_semantic_match_columns_with_llm_invalid_json(self, mock_call_llm):
        """Test fallback behavior when call_llm returns invalid JSON string."""
        mock_call_llm.return_value = "invalid json {"
        result = app.semantic_match_columns_with_llm(
            ['col1'], ['col2'], 'OpenAI', 'url', 'model'
        )
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
