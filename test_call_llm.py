import sys
import os
import unittest
from unittest.mock import patch, MagicMock

def create_streamlit_mock():
    st_mock = MagicMock()

    def mock_cache_resource(*args, **kwargs):
        def wrapper(func):
            func.clear = MagicMock()
            return func
        if len(args) == 1 and callable(args[0]):
            args[0].clear = MagicMock()
            return args[0]
        return wrapper

    def mock_cache_data(*args, **kwargs):
        def wrapper(func):
            func.clear = MagicMock()
            return func
        if len(args) == 1 and callable(args[0]):
            args[0].clear = MagicMock()
            return args[0]
        return wrapper

    mock_cache_resource.clear = MagicMock()
    mock_cache_data.clear = MagicMock()

    st_mock.cache_resource = mock_cache_resource
    st_mock.cache_data = mock_cache_data

    def mock_columns(n, *args, **kwargs):
        if isinstance(n, (list, tuple)):
            return [MagicMock() for _ in n]
        return [MagicMock() for _ in range(n)]

    st_mock.columns = mock_columns
    st_mock.sidebar = MagicMock()
    st_mock.sidebar.columns = mock_columns
    st_mock.session_state = {}
    st_mock.secrets = {}

    return st_mock

class TestCallLLM(unittest.TestCase):
    def setUp(self):
        os.environ['KAGGLE_USERNAME'] = 'dummy'
        os.environ['KAGGLE_KEY'] = 'dummy'

        self.mock_modules = {
            'streamlit': create_streamlit_mock(),
            'pandas': MagicMock(),
            'numpy': MagicMock(),
            'scipy': MagicMock(),
            'scipy.stats': MagicMock(),
            'thefuzz': MagicMock(),
            'plotly': MagicMock(),
            'plotly.express': MagicMock(),
            'plotly.graph_objects': MagicMock(),
            'PyPDF2': MagicMock(),
            'docx': MagicMock(),
            'requests': MagicMock()
        }
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    @patch('app.requests.post')
    def test_anthropic(self, mock_post):
        from app import call_llm

        mock_response = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Anthropic response"}]}
        mock_post.return_value = mock_response

        result = call_llm(
            prompt="Hello",
            llm_provider="Anthropic",
            llm_url="http://anthropic.com",
            model_name="claude",
            api_key="key",
            proxies=None,
            system_prompt="System"
        )

        self.assertEqual(result, "Anthropic response")
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://anthropic.com/messages")
        self.assertEqual(mock_post.call_args[1]["json"]["system"], "System")

    @patch('app.requests.post')
    def test_gemini(self, mock_post):
        from app import call_llm

        mock_response = MagicMock()
        mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}]}
        mock_post.return_value = mock_response

        result = call_llm(
            prompt="Hello",
            llm_provider="Gemini",
            llm_url="http://gemini.com",
            model_name="gemini-pro",
            api_key="key",
            proxies=None,
            system_prompt="System"
        )

        self.assertEqual(result, "Gemini response")
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://gemini.com/models/gemini-pro:generateContent?key=key")
        self.assertEqual(mock_post.call_args[1]["json"]["systemInstruction"]["parts"][0]["text"], "System")

    @patch('app.requests.post')
    def test_openai(self, mock_post):
        from app import call_llm

        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OpenAI response"}}]}
        mock_post.return_value = mock_response

        result = call_llm(
            prompt="Hello",
            llm_provider="OpenAI",
            llm_url="http://openai.com",
            model_name="gpt-4",
            api_key="key",
            proxies=None,
            system_prompt="System"
        )

        self.assertEqual(result, "OpenAI response")
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://openai.com/chat/completions")
        self.assertEqual(mock_post.call_args[1]["json"]["messages"][0]["content"], "System")

    @patch('app.requests.post')
    def test_ollama(self, mock_post):
        from app import call_llm

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_post.return_value = mock_response

        result = call_llm(
            prompt="Hello",
            llm_provider="Ollama",
            llm_url="http://ollama.com",
            model_name="llama2",
            api_key="key",
            proxies={"http": "http://proxy:8080"},
            system_prompt="System"
        )

        self.assertEqual(result, "Ollama response")
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://ollama.com/api/generate")
        self.assertEqual(mock_post.call_args[1]["json"]["system"], "System")
        self.assertEqual(mock_post.call_args[1]["proxies"], {"http": "http://proxy:8080"})

    @patch('app.requests.post')
    def test_http_error(self, mock_post):
        from app import call_llm

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_post.return_value = mock_response

        with self.assertRaises(Exception):
            call_llm(
                prompt="Hello",
                llm_provider="Anthropic",
                llm_url="http://anthropic.com",
                model_name="claude",
                api_key="key",
                proxies=None
            )

if __name__ == '__main__':
    unittest.main()
