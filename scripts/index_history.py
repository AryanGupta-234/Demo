from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.history_index import index_history
from pre_cab.persistent_memory import SQLiteUnifiedMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent Normal-CR historical memory index")
    parser.add_argument("input", type=Path)
    parser.add_argument("--memory-db", type=Path, default=Path(".pre_cab/history.sqlite3"))
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must be a JSON list")
    memory = SQLiteUnifiedMemory(args.memory_db)
    count = index_history(memory, records)
    print(json.dumps({"indexed_normal_crs": count, "memory_db": str(args.memory_db)}, indent=2))


if __name__ == "__main__":
    main()
