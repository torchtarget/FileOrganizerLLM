import importlib
import types
from unittest.mock import MagicMock

# Patch embeddings to avoid network
mock_embeddings = types.ModuleType("langchain_huggingface")
mock_embed_instance = MagicMock()
mock_embed_instance.embed_query.return_value = [1.0]
mock_embeddings.HuggingFaceEmbeddings = MagicMock(return_value=mock_embed_instance)

import sys

sys.modules["langchain_huggingface"] = mock_embeddings
sys.modules["langchain_huggingface.embeddings"] = mock_embeddings

mapper = importlib.import_module("file_organizer.mapper")
mapper.get_embedding = MagicMock(return_value=[1.0])
mapper.extract_text_file = MagicMock(return_value="test")


def test_cosine_similarity():
    assert mapper.cosine_similarity([1, 0], [1, 0]) == 1
    assert mapper.cosine_similarity([1, 0], [0, 1]) == 0


def test_suggest_folder_for_file(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("data")
    vectors = {str(tmp_path / "a"): [1.0], str(tmp_path / "b"): [0.0]}
    dest = mapper.suggest_folder_for_file(str(file), vectors)
    assert dest == str(tmp_path / "a")
