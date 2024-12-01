# azure_openai.py
from typing import Iterable, List
from openai import AsyncAzureOpenAI
from openai.types import EmbeddingModel

from .client import EmbedderClient, EmbedderConfig

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-large'

class AzureOpenAIEmbedderConfig(EmbedderConfig):
    """Azure OpenAI Embedder Config"""
    embedding_model: EmbeddingModel | str = DEFAULT_EMBEDDING_MODEL
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    azure_deployment: str | None = None

class AzureOpenAIEmbedder(EmbedderClient):
    """Azure OpenAI Embedder Client"""

    def __init__(self, config: AzureOpenAIEmbedderConfig | None = None):
        if config is None:
            config = AzureOpenAIEmbedderConfig()
            
        if not isinstance(config, AzureOpenAIEmbedderConfig):
            raise TypeError("config must be instance of AzureOpenAIEmbedderConfig") 
            
        self.config = config
        self.client = AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment or config.embedding_model
        )

    async def create(
        self, 
        input_data: str | List[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        result = await self.client.embeddings.create(
            input=input_data,
            model=self.config.azure_deployment or self.config.embedding_model
        )
        return result.data[0].embedding[: self.config.embedding_dim]