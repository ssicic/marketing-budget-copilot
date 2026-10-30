"""Local DuckDB warehouse: the single source of data for training and serving."""

from mbc.warehouse.db import DB_PATH, build, connect, read_sql

__all__ = ["DB_PATH", "build", "connect", "read_sql"]
