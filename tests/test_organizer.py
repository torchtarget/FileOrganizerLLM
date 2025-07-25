import os
import sys
import types
import importlib
from unittest.mock import MagicMock

# Patch HuggingFaceEmbeddings before importing organizer to avoid network calls
mock_embeddings = types.ModuleType("langchain_community.embeddings")
mock_embed_instance = MagicMock()
mock_embed_instance.embed_query.return_value = [0.0]
mock_embeddings.HuggingFaceEmbeddings = MagicMock(return_value=mock_embed_instance)

sys.modules['langchain_community.embeddings'] = mock_embeddings

organizer = importlib.import_module('file_organizer.organizer')
organizer.get_llm = MagicMock(return_value=MagicMock())


def test_get_display_path(tmp_path):
    root = tmp_path / "root"
    sub = root / "sub" / "child"
    sub.mkdir(parents=True)
    org = organizer.FolderOrganizer(str(root))
    path = org.get_display_path(str(sub))
    assert path == os.path.join(root.name, "sub", "child")


def test_build_folder_tree_and_order(tmp_path):
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "c").mkdir(parents=True)
    org = organizer.FolderOrganizer(str(root))
    tree = org.build_tree()
    order = org.order
    assert order[-1] == str(root)
    for folder in tree[str(root)]:
        assert order.index(folder) < order.index(str(root))


def test_extract_text_file_txt(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello world")
    assert organizer.extract_text_file(str(file_path)) == "hello world"
    assert organizer.extract_text_file(str(file_path), 5) == "hello"


def test_get_sample_files(tmp_path):
    root = tmp_path
    paths = []
    for i in range(3):
        p = root / f"f{i}.txt"
        p.write_text("data")
        os.utime(p, (p.stat().st_atime - i, p.stat().st_mtime - i))
        paths.append(p)
    selected = organizer.get_sample_files(str(root), 2)
    assert len(selected) == 2
    assert set(selected).issubset({p.name for p in paths})
