from __future__ import annotations

from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    in_progress = root / "modules" / "in_progress"
    completed = root / "modules" / "completed"

    if not in_progress.exists():
        print(f"Missing: {in_progress}")
        return 2

    md_files = sorted(p for p in in_progress.glob("*.md") if p.is_file())
    if not md_files:
        print("No in-progress module notes found.")
        return 0

    total_unchecked = 0
    ready: list[Path] = []

    for path in md_files:
        text = path.read_text()
        unchecked = sum(1 for line in text.splitlines() if line.lstrip().startswith("- [ ]"))
        total_unchecked += unchecked
        status = "READY" if unchecked == 0 else f"{unchecked} unchecked"
        print(f"{path.name}: {status}")
        if unchecked == 0:
            ready.append(path)

    if ready:
        print("")
        print("Modules ready to move to completed/:")
        for p in ready:
            print(f"  - {p.name}")
        print(f"(target dir: {completed})")

    return 0 if total_unchecked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
