"""Build the DuckDB warehouse from the simulators.

    python -m mbc.warehouse
"""

from mbc.warehouse.db import DB_PATH, build


def main() -> None:
    counts = build()
    print(f"Built {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:<22} {n:>6} rows")


if __name__ == "__main__":
    main()
