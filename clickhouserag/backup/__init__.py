from enum import Enum


class BackupFormat(Enum):
    JSON = "json"
    PARQUET = "parquet"
    PROTOBYTE = "protobyte"
    CSV = "csv"
    EXCEL = "excel"
