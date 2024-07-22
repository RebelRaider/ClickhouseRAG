# ClickhouseRAG

ClickhouseRAG is a Python package designed for efficient data access and management in Clickhouse. It provides an easy-to-use interface for connecting to Clickhouse, executing queries, and managing tables with support for Vectorizers and Retrieval-Augmented Generation (RAG) operations.

## Features

- **Easy Clickhouse Connection**: Seamlessly connect to your Clickhouse database.
- **Table Management**: Effortlessly manage tables with CRUD operations.
- **Vectorization**: Integrate with vectorizers for text and data embedding.
- **RAG Operations**: Perform Retrieval-Augmented Generation tasks.
- **Backup and Restore**: Backup your database to a file and restore it easily.
- **Cosine Similarity Search**: Search data based on cosine similarity.

## Installation

You can install ClickhouseRAG via pip:

```sh
pip install clickhouserag
```

## Usage

### Connecting to Clickhouse

Create a client to connect to your Clickhouse database.

```python
from clickhouserag.data_access.clickhouse_client import ClickhouseConnectClient

client = ClickhouseConnectClient(
    host="localhost",
    port=9000,
    username="default",
    password="",
    database="default"
)
client.connect()
```

### Defining Table Schema

Define the schema for your table in Clickhouse.

```python
table_schema = {
    "id": "UInt32",
    "title": "String",
    "vector": "Array(Float64)"
}
```

### Managing Tables

Create an instance of `RAGManager` to manage your table with the specified engine and schema.

```python
from clickhouserag.rag.manager import RAGManager

rag_manager = RAGManager(client, "rag_table", table_schema, engine="MergeTree", order_by="id")
```

### Creating and Adding Vectorizer

Create and add a Transformers vectorizer to the RAGManager.

```python
import torch
from transformers import AutoModel, AutoTokenizer
from clickhouserag.vectorizers.base import VectorizerBase

class TransformersVectorizer(VectorizerBase):
    """Vectorizer that uses a Transformers model to convert text to vectors."""
    
    def __init__(self, model_name: str) -> None:
        """Initialize the TransformersVectorizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def vectorize(self, data: Any) -> List[float]:
        """Convert text data into a vector representation using a Transformers model."""
        if not isinstance(data, str):
            raise ValueError("Data should be a string for text vectorization.")
        
        inputs = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        return vector

    def bulk_vectorize(self, data: Any) -> List[List[float]]:
        """Convert listed text data into a vector representation using a Transformers model."""

        if not isinstance(data, List[str]):
            raise ValueError("Data should be a list of a strings for text vectorization.")

        inputs = self.tokenizer(
            data, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        return vector

transformers_vectorizer = TransformersVectorizer(model_name="distilbert-base-uncased")
rag_manager.add_vectorizer("transformers", transformers_vectorizer)
```

### Adding Data with Vectorization

Add individual data records with vectorization through Transformers.

```python
data = {"id": 1, "title": "Sample text data for transformers"}
rag_manager.add_data(data, vectorizer_name="transformers")
```

### Bulk Adding Data with Vectorization

Add multiple data records with vectorization through Transformers.

```python
bulk_data = [
    {"id": 2, "title": "Sample text data 1 for transformers"},
    {"id": 3, "title": "Sample text data 2 for transformers"},
    {"id": 4, "title": "Sample text data 3 for transformers"}
]
rag_manager.add_bulk_data(bulk_data, vectorizer_name="transformers")
```

### Retrieving Data by ID

Retrieve data from the RAG by ID.

```python
data = rag_manager.get_data(1)
print("Data with ID 1:", data)
```

### Updating Data with Vectorization

Update data with vectorization through Transformers.

```python
updated_data = {"id": 1, "title": "Updated text data for transformers"}
rag_manager.update_data(1, updated_data, vectorizer_name="transformers")
```

### Executing Text Search

Perform a text search on the RAG.

```python
query = "SELECT * FROM rag_table WHERE title LIKE '%Sample%'"
search_results = rag_manager.search(query)
print("Search results:", search_results)
```

### Executing Cosine Similarity Search

Perform a cosine similarity search on the RAG.

```python
import numpy as np

embedding = np.random.rand(768)  # Example random vector
similarity_results = rag_manager.similarity_search(embedding, top_k=2, columns=["id", "title"])
print("Similarity search results:", similarity_results)
```

### Deleting Data

Delete data from the RAG by ID.

```python
rag_manager.delete_data(1)
```

### Backing Up the Database

Backup the database to a JSON file.

```python
rag_manager.backup_database("backup.json")
```

### Resetting and Restoring the Database

Reset and restore the database from a backup file.

```python
rag_manager.reset_database()
rag_manager.restore_database("backup.json", table_schema=table_schema)
```

### Closing the Database Connection

Close the connection to the Clickhouse database.

```python
client.close()
```

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact Leonid Chesnikov at <leonid.chesnikov@gmail.com>.

## Project Structure

- **clickhouserag.data_access**: Contains modules for managing Clickhouse connections and tables.
- **clickhouserag.rag**: Contains modules for RAG operations and vectorizers.

## Requirements

- `clickhouse-driver`
- `numpy`

These dependencies are automatically installed when you install the package via pip.

## Development

To contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

We appreciate your contributions and efforts in improving this project!

## Keywords

- Clickhouse
- Data Access
- Table Management
- Vectorizer
- RAG (Retrieval-Augmented Generation)

---

[GitHub Repository](https://github.com/RebelRaider/ClickhouseRAG)
