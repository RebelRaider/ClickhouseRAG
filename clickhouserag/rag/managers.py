"""RAG manager module for managing data in Clickhouse."""

from typing import Dict, Optional

from clickhouserag.clickhouse.clients import ClickhouseClient
from clickhouserag.rag.base import RAGBase
from clickhouserag.rag.mixins import (
    BackupMixin,
    DataOperationsMixin,
    TableManagerMixin,
    VectorizerMixin,
)


class RAGManager(TableManagerMixin, DataOperationsMixin, BackupMixin, VectorizerMixin, RAGBase):
    """Manager class for RAG operations in Clickhouse."""

    def __init__(self, client: ClickhouseClient, table_name: str, table_schema: Optional[Dict[str, str]] = None, engine: str = "MergeTree", order_by: str = "id") -> None:
        TableManagerMixin.__init__(self, client, table_name, table_schema, engine, order_by)
        BackupMixin.__init__(self, client, self.table_manager)
        VectorizerMixin.__init__(self)
        RAGBase.__init__(self)
