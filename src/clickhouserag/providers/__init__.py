"""Composable model adapters for local inference and hosted services."""

from clickhouserag._structured import JsonModel, ModelResponseError, StructuredLLM
from clickhouserag.providers._http import ModelHTTPError
from clickhouserag.providers.anthropic import AnthropicMessages
from clickhouserag.providers.bedrock import BedrockConverse, BedrockTitanEmbeddings
from clickhouserag.providers.cohere import CohereChat, CohereEmbeddings
from clickhouserag.providers.gemini import GeminiChat, GeminiEmbeddings
from clickhouserag.providers.ollama import Ollama, OllamaChat, OllamaEmbeddings
from clickhouserag.providers.openai_compatible import (
    OpenAICompatibleChat,
    OpenAICompatibleEmbeddings,
    OpenAICompatibleResponses,
)
from clickhouserag.providers.rerank import CohereReranker, TEIReranker, VoyageReranker
from clickhouserag.providers.vertex import VertexGeminiChat, VertexTextEmbeddings
from clickhouserag.providers.voyage import VoyageEmbeddings

__all__ = [
    "AnthropicMessages",
    "BedrockConverse",
    "BedrockTitanEmbeddings",
    "CohereChat",
    "CohereEmbeddings",
    "CohereReranker",
    "GeminiChat",
    "GeminiEmbeddings",
    "JsonModel",
    "ModelHTTPError",
    "ModelResponseError",
    "Ollama",
    "OllamaChat",
    "OllamaEmbeddings",
    "OpenAICompatibleChat",
    "OpenAICompatibleEmbeddings",
    "OpenAICompatibleResponses",
    "StructuredLLM",
    "TEIReranker",
    "VertexGeminiChat",
    "VertexTextEmbeddings",
    "VoyageEmbeddings",
    "VoyageReranker",
]
