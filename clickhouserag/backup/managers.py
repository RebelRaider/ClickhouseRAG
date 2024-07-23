import logging
from typing import Dict, Optional

import pandas as pd

from clickhouserag.backup import BackupFormat
from clickhouserag.clickhouse.base import ClickhouseClient
from clickhouserag.clickhouse.managers import ClickhouseTableManager
from clickhouserag.utils import (
    check_installed,
    get_format_from_path,
    load_json,
    save_json,
)


class BackupManager:
    """Class to manage backup operations for RAG data."""

    def __init__(self, client: ClickhouseClient, table_manager: ClickhouseTableManager) -> None:
        """Initialize the BackupManager.

        Args:
        ----
            client (ClickhouseClient): The Clickhouse client.
            table_manager (ClickhouseTableManager): The table manager instance.
        """
        self.client = client
        self.table_manager = table_manager
        self.logger = logging.getLogger(__name__)

        self._backup_handlers = {
            BackupFormat.JSON: self._backup_to_json,
            BackupFormat.PARQUET: self._backup_to_parquet,
            BackupFormat.PROTOBYTE: self._backup_to_protobyte,
            BackupFormat.CSV: self._backup_to_csv,
            BackupFormat.EXCEL: self._backup_to_excel,
        }

        self._restore_handlers = {
            BackupFormat.JSON: self._restore_from_json,
            BackupFormat.PARQUET: self._restore_from_parquet,
            BackupFormat.PROTOBYTE: self._restore_from_protobyte,
            BackupFormat.CSV: self._restore_from_csv,
            BackupFormat.EXCEL: self._restore_from_excel,
        }

    def backup_to_file(self, path: str) -> None:
        """Backup the RAG database to a file.

        Args:
        ----
            path (str): The file path to save the backup.

        Raises:
        ------
            ValueError: If an unsupported format is specified.
            RuntimeError: If the backup operation fails.
        """
        backup_format = get_format_from_path(path)
        handler = self._backup_handlers.get(BackupFormat(backup_format))
        if handler:
            handler(path)
        else:
            raise ValueError(f"Unsupported file extension for backup: {path}")

    def restore_from_file(self, path: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        """Restore the RAG database from a file.

        Args:
        ----
            path (str): The file path to restore the backup from.
            table_schema (Optional[Dict[str, str]]): The schema of the table.
            engine (str): The engine to use for the table.
            order_by (str): The column to order the table by.

        Raises:
        ------
            ValueError: If an unsupported format is specified.
            RuntimeError: If the restore operation fails.
        """
        backup_format = get_format_from_path(path)
        handler = self._restore_handlers.get(BackupFormat(backup_format))
        if handler:
            handler(path, table_schema, engine, order_by)
        else:
            raise ValueError(f"Unsupported file extension for restore: {path}")

    def _backup_to_json(self, path: str) -> None:
        data = self.table_manager.fetch_all()
        save_json(data, path)
        self.logger.info(f"Database backup created at {path} in JSON format")

    def _backup_to_parquet(self, path: str) -> None:
        check_installed("pyarrow", "fastparquet")
        data = self.table_manager.fetch_all()
        df = pd.DataFrame(data)
        df.to_parquet(path, engine="pyarrow")
        self.logger.info(f"Database backup created at {path} in Parquet format")

    def _backup_to_protobyte(self, path: str) -> None:
        check_installed("protobuf")
        # TODO Implement protobuf serialization
        raise ValueError("Protobuf serialization is not implemented yet")

    def _backup_to_csv(self, path: str) -> None:
        data = self.table_manager.fetch_all()
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        self.logger.info(f"Database backup created at {path} in CSV format")

    def _backup_to_excel(self, path: str) -> None:
        data = self.table_manager.fetch_all()
        df = pd.DataFrame(data)
        df.to_excel(path, index=False)
        self.logger.info(f"Database backup created at {path} in Excel format")

    def _restore_from_json(self, path: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        if table_schema:
            self._initialize_table(table_schema, engine, order_by)
        data = load_json(path)
        self.table_manager.reset_table()
        self.table_manager.insert(data)
        self.logger.info(f"Database restored from {path} in JSON format")

    def _restore_from_parquet(self, path: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        check_installed("pyarrow", "fastparquet")
        if table_schema:
            self._initialize_table(table_schema, engine, order_by)
        df = pd.read_parquet(path, engine="pyarrow")
        data = df.to_dict(orient="records")
        self.table_manager.reset_table()
        self.table_manager.insert(data)
        self.logger.info(f"Database restored from {path} in Parquet format")

    def _restore_from_protobyte(self, path: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        check_installed("protobuf")
        # TODO Implement protobuf deserialization
        raise ValueError("Protobuf deserialization is not implemented yet")

    def _restore_from_csv(self, path: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        if table_schema:
            self._initialize_table(table_schema, engine, order_by)
        df = pd.read_csv(path)
        data = df.to_dict(orient="records")
        self.table_manager.reset_table()
        self.table_manager.insert(data)
        self.logger.info(f"Database restored from {path} in CSV format")

    def _restore_from_excel(self, path: str, table_schema: Optional[Dict[str, str]], engine: str, order_by: str) -> None:
        if table_schema:
            self._initialize_table(table_schema, engine, order_by)
        df = pd.read_excel(path)
        data = df.to_dict(orient="records")
        self.table_manager.reset_table()
        self.table_manager.insert(data)
        self.logger.info(f"Database restored from {path} in Excel format")

    def _initialize_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        """Initialize the table in Clickhouse based on the provided schema if it does not exist."""
        try:
            if not self._check_table_exists():
                self._create_table(table_schema, engine, order_by)
            else:
                self.logger.info("Table already exists.")
        except Exception as e:
            self.logger.exception(f"Error initializing table: {e}")

    def _create_table(self, table_schema: Dict[str, str], engine: str, order_by: str) -> None:
        """Create a table in Clickhouse based on the provided schema."""
        try:
            fields = ", ".join([f"{name} {dtype}" for name, dtype in table_schema.items()])
            query = f"CREATE TABLE ({fields}) ENGINE = {engine} ORDER BY {order_by}"
            self.client.execute_query(query)
            self.logger.info(f"Table created with schema: {table_schema}, engine: {engine}, order by: {order_by}")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise RuntimeError("Failed to create table") from e

    def _check_table_exists(self) -> bool:
        """Check if the table exists in the database."""
        query = "EXISTS TABLE"
        try:
            result = self.client.execute_query(query)
            return result[0][0] == 1
        except Exception as e:
            self.logger.error(f"Failed to check if table exists: {e}")
            raise RuntimeError("Failed to check if table exists") from e
