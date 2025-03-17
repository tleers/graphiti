from dataclasses import dataclass
from typing import Optional

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0
DEFAULT_API_VERSION = "2024-02-15-preview"

@dataclass
class LLMConfig:
    """Base configuration for LLM API interactions"""
    api_key: Optional[str] = None
    model: Optional[str] = None 
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

@dataclass
class AzureLLMConfig(LLMConfig):
    """Azure-specific LLM configuration"""
    api_version: str = DEFAULT_API_VERSION
    azure_deployment: Optional[str] = None

    def __post_init__(self):
        """Validate Azure configuration"""
        if not self.api_version:
            raise ValueError("api_version required for Azure")
        if not (self.azure_deployment or self.model):
            raise ValueError("Either azure_deployment or model required")
        if not self.base_url or "azure" not in self.base_url.lower():
            raise ValueError("base_url must be an Azure endpoint")