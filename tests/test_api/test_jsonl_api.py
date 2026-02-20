from __future__ import annotations

from pathlib import Path

import pytest

from tess_vetter.api.jsonl import append_jsonl, iter_jsonl, stream_existing_candidate_keys


def test_iter_jsonl_streams_objects_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")

    rows = list(iter_jsonl(path))

    assert rows == [{"a": 1}, {"b": 2}]


def test_iter_jsonl_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":1}\n{not-json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Malformed JSON at .*:2"):
        list(iter_jsonl(path))


def test_iter_jsonl_rejects_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad_type.jsonl"
    path.write_text('[1,2,3]\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected object at .*:1"):
        list(iter_jsonl(path))


def test_stream_existing_candidate_keys_handles_missing_file_and_filters_values(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    assert stream_existing_candidate_keys(missing) == set()

    path = tmp_path / "existing.jsonl"
    path.write_text(
        '\n'.join(
            [
                '{"candidate_key":"TOI-1.01"}',
                '{"candidate_key":""}',
                '{"candidate_key":7}',
                '{"no_key":true}',
                '{"candidate_key":"TOI-2.01"}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    assert stream_existing_candidate_keys(path) == {"TOI-1.01", "TOI-2.01"}


def test_append_jsonl_appends_compact_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"

    append_jsonl(path, {"b": 2, "a": [1, 2]})
    append_jsonl(path, {"candidate_key": "TOI-3.01"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"b":2,"a":[1,2]}', '{"candidate_key":"TOI-3.01"}']
