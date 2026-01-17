from __future__ import annotations

import argparse
import json
from pathlib import Path

from bittr_tess_vetter.exofop.client import ExoFopClient
from bittr_tess_vetter.exofop.types import ExoFopSelectors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bittr-tess-vetter exofop-fetch",
        description="Fetch and cache ExoFOP-TESS artifacts for a TIC/TOI/name.",
    )
    p.add_argument("--cache-dir", required=True, help="Cache root directory.")
    p.add_argument("--cookie-jar", default=None, help="Optional Mozilla/Netscape cookie jar path.")
    p.add_argument("--target", required=True, help="Target identifier (TIC, TOI-xxx, name).")
    p.add_argument("--force-refresh", action="store_true", help="Refresh indices and archives.")
    p.add_argument("--index-only", action="store_true", help="Only download indices/summaries.")
    p.add_argument("--types", default=None, help="Comma-separated file Types to include (e.g. Image,Spectrum).")
    p.add_argument("--regex", default=None, help="Regex to filter filenames (applied to basename).")
    p.add_argument("--tag-ids", default=None, help="Comma-separated tag IDs to include.")
    p.add_argument("--max-files", type=int, default=None, help="Maximum number of files to fetch.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = ExoFopClient(
        cache_dir=Path(args.cache_dir),
        cookie_jar_path=Path(args.cookie_jar) if args.cookie_jar else None,
    )
    selectors = None
    if not args.index_only:
        types = {t.strip() for t in (args.types or "").split(",") if t.strip()} or None
        tag_ids = None
        if args.tag_ids:
            tag_ids = {int(x) for x in args.tag_ids.split(",") if x.strip()}
        selectors = ExoFopSelectors(
            types=types,
            filename_regex=args.regex,
            tag_ids=tag_ids,
            max_files=args.max_files,
        )
    result = client.fetch(
        target=args.target,
        selectors=selectors,
        include_summaries=True,
        force_refresh=bool(args.force_refresh),
    )
    print(json.dumps({"cache_root": str(result.cache_root), "manifest": str(result.manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
