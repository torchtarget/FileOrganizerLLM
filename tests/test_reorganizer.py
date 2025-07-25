import importlib
import types
import json
from unittest.mock import MagicMock

# Patch embeddings to avoid network
mock_embeddings = types.ModuleType("langchain_huggingface")
mock_embed_instance = MagicMock()
mock_embed_instance.embed_query.return_value = [1.0]
mock_embeddings.HuggingFaceEmbeddings = MagicMock(return_value=mock_embed_instance)

import sys

sys.modules["langchain_huggingface"] = mock_embeddings
sys.modules["langchain_huggingface.embeddings"] = mock_embeddings

reorganizer = importlib.import_module("file_organizer.reorganizer")
reorganizer.get_llm = MagicMock(return_value=MagicMock())
reorganizer.map_files = MagicMock(return_value={"/a/x.txt": "/b"})
reorganizer.apply_mapping = MagicMock()


def test_analyze(tmp_path):
    vectors = {str(tmp_path): [1.0], str(tmp_path / "b"): [1.0]}
    contexts = {str(tmp_path): "root", str(tmp_path / "b"): "b"}
    (tmp_path / "folder_vectors.json").write_text(json.dumps(vectors))
    (tmp_path / "folder_contexts.json").write_text(json.dumps(contexts))
    reg = reorganizer.Reorganizer(str(tmp_path))
    reg.suggest_merges = MagicMock(return_value=[(str(tmp_path), str(tmp_path / "b"))])
    reg.suggest_moves = MagicMock(return_value={"f": str(tmp_path / "b")})
    reg.analyze()
    plan = json.loads((tmp_path / "reorg_plan.json").read_text())
    assert plan["merge_candidates"]
    assert plan["move_suggestions"]


def test_apply(tmp_path):
    plan = {"move_suggestions": {"a": "b"}}
    (tmp_path / "reorg_plan.json").write_text(json.dumps(plan))
    reg = reorganizer.Reorganizer(str(tmp_path))
    reg.apply()
    reorganizer.apply_mapping.assert_called_once()
