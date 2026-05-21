import unittest
from unittest.mock import MagicMock, patch
import os

# Create an advanced mock for streamlit
st_mock = MagicMock()

# Mock decorators to return the original function
def mock_cache(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def decorator(func):
        return func
    return decorator

# Attach clear method
cache_func = MagicMock(side_effect=mock_cache)
cache_func.clear = MagicMock()

st_mock.cache_resource = cache_func
st_mock.cache_data = cache_func

def mock_columns(n):
    if isinstance(n, (list, tuple)):
        return [MagicMock() for _ in n]
    return [MagicMock() for _ in range(n)]

st_mock.columns = mock_columns

# Mock missing dependencies
mock_modules = {
    'streamlit': st_mock,
    'pandas': MagicMock(),
    'numpy': MagicMock(),
    'scipy': MagicMock(),
    'scipy.stats': MagicMock(),
    'requests': MagicMock(),
    'thefuzz': MagicMock(),
    'thefuzz.fuzz': MagicMock(),
    'thefuzz.process': MagicMock(),
    'plotly': MagicMock(),
    'plotly.express': MagicMock(),
    'plotly.graph_objects': MagicMock(),
    'PyPDF2': MagicMock(),
    'docx': MagicMock()
}

with patch.dict('sys.modules', mock_modules):
    os.environ['KAGGLE_USERNAME'] = 'dummy'
    os.environ['KAGGLE_KEY'] = 'dummy'
    from app import is_valid_dataset

class DummyDataset:
    def __init__(self, title=None, description=None, tags=None):
        if title is not None:
            self.title = title
        if description is not None:
            self.description = description
        if tags is not None:
            self.tags = tags

class TestIsValidDataset(unittest.TestCase):
    def test_default_valid_dataset(self):
        d = DummyDataset(title="Normal Data", description="Just some dataset")
        self.assertTrue(is_valid_dataset(d, "some context"))

    def test_rejects_image_dataset_without_user_intent(self):
        d = DummyDataset(title="Computer Vision Data", description="Contains pixels and resnet")
        self.assertFalse(is_valid_dataset(d, "I want to analyze some telemetry"))

    def test_accepts_image_dataset_with_user_intent(self):
        d = DummyDataset(title="Computer Vision Data", description="Contains pixels and resnet")
        self.assertTrue(is_valid_dataset(d, "I need a computer vision dataset with image"))

    def test_accepts_image_keyword_if_tabular_keywords_present(self):
        d = DummyDataset(title="CNN Results", description="tabular csv file for cnn performance")
        self.assertTrue(is_valid_dataset(d, "some context"))

    def test_handles_missing_attributes(self):
        d = DummyDataset()
        self.assertTrue(is_valid_dataset(d, "some context"))

    def test_handles_none_attributes(self):
        d = DummyDataset()
        d.title = None
        d.description = None
        d.tags = None
        self.assertTrue(is_valid_dataset(d, "some context"))

    def test_handles_none_user_context(self):
        d = DummyDataset(title="Some Data", description="Data")
        self.assertTrue(is_valid_dataset(d, None))

    def test_with_tags(self):
        d = DummyDataset(title="Data", description="Data", tags=["tabular", "csv"])
        self.assertTrue(is_valid_dataset(d, "context"))

    def test_rejects_with_tags(self):
        d = DummyDataset(title="Data", description="Data", tags=["image dataset", "cnn"])
        self.assertFalse(is_valid_dataset(d, "context"))

    def test_none_tags(self):
        d = DummyDataset(title="Data", description="Data", tags=[None])
        self.assertTrue(is_valid_dataset(d, "context"))

if __name__ == "__main__":
    unittest.main()
