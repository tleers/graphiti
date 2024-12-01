# azure_reranker_client.py
import asyncio
import logging
import typing
from openai import AsyncAzureOpenAI
import openai

from .client import CrossEncoderClient
from ..llm_client import LLMConfig, RateLimitError  # Same imports as OpenAI client
from ..prompts import Message
from ..llm_client.config import AzureLLMConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

class AzureOpenAIRerankerClient(CrossEncoderClient):
    def __init__(
        self,
        config: AzureLLMConfig | None = None,
        client: typing.Any = None
    ):
        if config is None:
            config = AzureLLMConfig()
            
        if not isinstance(config, AzureLLMConfig):
            raise TypeError("config must be instance of AzureLLMConfig")

        if client is None:
            self.client = AsyncAzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.base_url,
                api_version=config.api_version,
                azure_deployment=config.azure_deployment or config.model or DEFAULT_MODEL
            )
        else:
            self.client = client

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Rank passages using Azure OpenAI"""
        openai_messages_list: typing.Any = [
            [
                Message(
                    role='system',
                    content='You are an expert tasked with determining whether the passage is relevant to the query',
                ),
                Message(
                    role='user',
                    content=f"""
                           Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise. 
                           <PASSAGE>
                           {passage}
                           </PASSAGE>
                           <QUERY>
                           {query}
                           </QUERY>
                           """,
                ),
            ]
            for passage in passages
        ]

        try:
            responses = await asyncio.gather(
                *[
                    self.client.chat.completions.create(
                        model=self.client.azure_deployment,
                        messages=messages,
                        temperature=0,
                        max_tokens=1,
                        logit_bias={'6432': 1, '7983': 1},
                        logprobs=True,
                        top_logprobs=2,
                    )
                    for messages in openai_messages_list
                ]
            )

            responses_top_logprobs = [
                response.choices[0].logprobs.content[0].top_logprobs
                if response.choices[0].logprobs is not None
                and response.choices[0].logprobs.content is not None
                else []
                for response in responses
            ]

            scores: list[float] = []
            for top_logprobs in responses_top_logprobs:
                for logprob in top_logprobs:
                    if bool(logprob.token):
                        scores.append(logprob.logprob)

            results = [(passage, score) for passage, score in zip(passages, scores)]
            results.sort(reverse=True, key=lambda x: x[1])
            return results

        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise