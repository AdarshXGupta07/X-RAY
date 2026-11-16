"""Tests for the paper asset builder's table plumbing (no real results needed)."""

import pandas as pd

from tbshift.experiments.paper_assets import _md, fill_tables


def test_markdown_table_shape():
    out = _md(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert out.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |", "| 2 | y |"]


def test_fill_tables_replaces_only_marked_blocks_and_is_idempotent(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text("intro\n<!-- table:t1 -->\n<!-- /table:t1 -->\nmiddle\n"
                     "<!-- table:t2 -->\nold row\n<!-- /table:t2 -->\nend\n", encoding="utf-8")
    tables = {"t1": "| new |", "t2": "| two |\n|---|", "t9": "| unused |", "t3": None}
    assert fill_tables(paper, tables) == ["t1", "t2"]
    once = paper.read_text(encoding="utf-8")
    assert "old row" not in once and "| new |" in once and "| unused |" not in once
    assert once.startswith("intro\n") and "middle\n" in once and once.endswith("end\n")
    fill_tables(paper, tables)
    assert paper.read_text(encoding="utf-8") == once


def test_fill_tables_without_paper_does_nothing(tmp_path):
    assert fill_tables(tmp_path / "missing.md", {"t1": "x"}) == []
