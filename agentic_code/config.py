"""
Configuration management for AgenticCode using pydantic-settings.
Supports Google Gemini models, OpenAI, Anthropic, and local mock testing.
"""

import os
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_GLOBAL_GEMINI_CLIENT: Optional[Any] = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (or GOOGLE_API_KEY)",
    )
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Alternative Google API key environment variable",
    )
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    GITHUB_TOKEN: Optional[str] = Field(default=None, description="GitHub token (PAT or GitHub App token)")
    GITHUB_WEBHOOK_SECRET: Optional[str] = Field(default="dev-webhook-secret", description="HMAC secret for webhooks")

    # LLM Provider and Models (Defaulted to Google Gemini)
    DEFAULT_LLM_PROVIDER: Literal["gemini", "openai", "anthropic", "mock"] = Field(
        default="gemini", description="Default provider: gemini | openai | anthropic | mock"
    )
    DEFAULT_MODEL: str = Field(default="gemini-2.5-flash", description="Default model name")
    CODE_READER_MODEL: str = Field(default="gemini-2.5-flash", description="Model for Agent 01 Code Reader")
    PLANNER_MODEL: str = Field(default="gemini-2.5-flash", description="Model for Agent 02 Planner")
    TEST_WRITER_MODEL: str = Field(default="gemini-2.5-flash", description="Model for Agent 04 Test Writer")
    CODE_WRITER_MODEL: str = Field(default="gemini-2.5-flash", description="Model for Agent 03 Code Writer")
    TEMPERATURE: float = Field(default=0.0, description="LLM temperature")
    USE_GEMINI_CHAT_AFC: bool = Field(
        default=True,
        description="Enable Google Solution A: Use stateful Chat.send_message for Automatic Function Calling (AFC)",
    )

    # Hardened Sandbox Settings
    SANDBOX_IMAGE: str = Field(default="agentic-code-sandbox:latest", description="Docker image tag for sandbox")
    SANDBOX_USER: str = Field(default="1000:1000", description="UID:GID for non-root execution")
    SANDBOX_TIMEOUT_SECONDS: int = Field(default=60, description="Strict execution timeout in seconds")
    SANDBOX_MEM_LIMIT: str = Field(default="4096m", description="Container memory limit")
    SANDBOX_CPU_QUOTA: int = Field(default=200000, description="2.0 vCPUs (200000 microseconds / 100000 period)")
    SANDBOX_NETWORK_MODE: str = Field(default="none", description="Container network isolation mode")
    ALLOW_LOCAL_FALLBACK_SANDBOX: bool = Field(
        default=True,
        description="Allow isolated local execution when Docker engine is unreachable",
    )

    # State Machine & Repair Loops
    MAX_ITERATIONS: int = Field(default=3, description="Maximum iterative repair loops (k_max)")
    WORK_DIR: str = Field(default="./workspace_runs", description="Root directory for ephemeral clones")

    # Observability & Tracing
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="AgenticCode")
    OTEL_EXPORTER_ENABLED: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    def get_llm(
        self,
        model_name: Optional[str] = None,
        provider: Optional[Literal["gemini", "openai", "anthropic", "mock"]] = None,
        temperature: Optional[float] = None,
    ) -> BaseChatModel:
        """
        Instantiate and return a ChatModel based on provider and model configurations.
        Defaults to Google Gemini (e.g. gemini-2.5-flash, gemini-2.5-pro).
        """
        prov = provider or self.DEFAULT_LLM_PROVIDER
        mod = model_name or self.DEFAULT_MODEL
        temp = self.TEMPERATURE if temperature is None else temperature

        if prov == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = (
                self.GEMINI_API_KEY
                or self.GOOGLE_API_KEY
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or "dummy_key_for_testing"
            )
            return ChatGoogleGenerativeAI(
                model=mod,
                google_api_key=api_key,
                temperature=temp,
            )
        elif prov == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model_name=mod,
                anthropic_api_key=self.ANTHROPIC_API_KEY or "dummy_key_for_testing",
                temperature=temp,
            )
        elif prov == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model_name=mod,
                openai_api_key=self.OPENAI_API_KEY or "dummy_key_for_testing",
                temperature=temp,
            )
        else:
            # Fallback mock/testing model
            from langchain_core.language_models.fake_chat_models import FakeListChatModel

            return FakeListChatModel(
                responses=[
                    "Mock LLM Response: Analysis complete.",
                    "Mock LLM Response: Fix applied.",
                ]
            )

    def get_gemini_client(self) -> Any:
        """
        Returns an authenticated google.genai.Client instance with persistent lifecycle.
        """
        global _GLOBAL_GEMINI_CLIENT
        if _GLOBAL_GEMINI_CLIENT is not None:
            return _GLOBAL_GEMINI_CLIENT

        from google import genai

        api_key = (
            self.GEMINI_API_KEY
            or self.GOOGLE_API_KEY
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or "dummy_key_for_testing"
        )
        _GLOBAL_GEMINI_CLIENT = genai.Client(api_key=api_key)
        return _GLOBAL_GEMINI_CLIENT

    def create_gemini_chat(
        self,
        model_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Implements Google Solution A: Creates a stateful Chat session with Automatic Function Calling (AFC).
        Uses Chat.send_message instead of Models.generate_content for robust tool execution.
        """
        from google.genai import types

        client = self.get_gemini_client()
        mod = model_name or self.DEFAULT_MODEL
        temp = self.TEMPERATURE if temperature is None else temperature

        config_kwargs: Dict[str, Any] = {"temperature": temp}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = tools

        config = types.GenerateContentConfig(**config_kwargs)
        chat = client.chats.create(model=mod, config=config)
        chat._persistent_client = client
        return chat


# Global settings instance
settings = Settings()
