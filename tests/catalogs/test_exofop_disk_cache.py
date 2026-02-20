from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tess_vetter.platform.catalogs.exofop_target_page import fetch_exofop_target_summary
from tess_vetter.platform.catalogs.exofop_toi_table import fetch_exofop_toi_table


def test_exofop_toi_table_uses_disk_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "exofop" / "toi_table.pipe"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "tic_id|toi|tfopwg_disposition|comments\n1|123.01|PC|\n", encoding="utf-8"
    )

    with patch("tess_vetter.platform.catalogs.exofop_toi_table.requests.get") as get:
        get.side_effect = RuntimeError("network should not be called")
        table = fetch_exofop_toi_table(cache_ttl_seconds=999999, disk_cache_dir=tmp_path)

    assert table.entries_for_tic(1)


def test_exofop_target_page_uses_disk_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "exofop" / "target_pages" / "42.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        '{"tic_id": 42, "url": "x", "fetched_at_unix": 0, "grid_badges": {"Files": 7}, "followup_counts": {"files": 7}, "flags": []}',
        encoding="utf-8",
    )

    with patch("tess_vetter.platform.catalogs.exofop_target_page.requests.get") as get:
        get.side_effect = RuntimeError("network should not be called")
        summary = fetch_exofop_target_summary(
            tic_id=42, cache_ttl_seconds=999999, disk_cache_dir=tmp_path
        )

    assert summary.tic_id == 42
    assert summary.followup_counts["files"] == 7


def test_disk_cache_respects_ttl(tmp_path: Path) -> None:
    cache_path = tmp_path / "exofop" / "toi_table.pipe"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("tic_id|toi\n1|123.01\n", encoding="utf-8")

    # TTL=0 disables disk cache and should try network (we intercept as a clear signal).
    with patch("tess_vetter.platform.catalogs.exofop_toi_table.requests.get") as get:
        get.side_effect = RuntimeError("network called")
        with pytest.raises(RuntimeError, match="network called"):
            fetch_exofop_toi_table(cache_ttl_seconds=0, disk_cache_dir=tmp_path)
