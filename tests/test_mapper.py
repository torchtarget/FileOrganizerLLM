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
    contexts = {str(tmp_path / "a"): "A", str(tmp_path / "b"): "B"}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="1")
    dest = mapper.suggest_folder_for_file(
        str(file), vectors, contexts, llm, top_n=2
    )
    assert dest == str(tmp_path / "a")


def test_suggest_folder_for_file_min_similarity(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("data")
    vectors = {str(tmp_path / "a"): [0.0]}
    contexts = {str(tmp_path / "a"): "A"}
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="1")
    dest = mapper.suggest_folder_for_file(
        str(file), vectors, contexts, llm, top_n=1, min_similarity=0.5
    )
    assert dest == ""


def test_map_files_uses_existing_mapping(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("data")
    f2 = tmp_path / "b.txt"
    f2.write_text("data")

    existing = {str(f1): "/dest"}
    mapper.suggest_folder_for_file = MagicMock(return_value="/new")
    llm = MagicMock()

    mapping = mapper.map_files(
        str(tmp_path),
        {},
        {},
        llm,
        existing_mapping=existing,
    )

    assert mapping[str(f1)] == "/dest"
    assert mapping[str(f2)] == "/new"
    mapper.suggest_folder_for_file.assert_called_once_with(
        str(f2), {}, {}, llm, top_n=3, min_similarity=0.0
    )
