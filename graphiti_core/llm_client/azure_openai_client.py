"""azure_openai_client.py"""
import typing
from openai import AsyncAzureOpenAI

from .openai_client import OpenAIClient, DEFAULT_MODEL
from .config import AzureLLMConfig, LLMConfig
from ..prompts.models import Message

class AzureOpenAIClient(OpenAIClient):
    """AzureOpenAIClient extends OpenAIClient to support Azure OpenAI deployments."""

    def __init__(
        self, 
        config: AzureLLMConfig | None = None, 
        cache: bool = False, 
        client: typing.Any = None
    ):
        if config is None:
            config = AzureLLMConfig()

        if not isinstance(config, AzureLLMConfig):
            raise TypeError("config must be instance of AzureLLMConfig")

        super().__init__(config, cache)

        if client is None:
            self.client = AsyncAzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.base_url,
                api_version=config.api_version,
                azure_deployment=config.azure_deployment or config.model,
            )
        else:
            self.client = client

    async def _generate_response(self, messages: list[Message]) -> dict[str, typing.Any]:
        """No changes needed to response generation - inherits from OpenAIClient"""
        return await super()._generate_response(messages)