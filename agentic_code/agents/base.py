"""
Base Persona abstraction for AgenticCode specialized agents.
"""

import json
import re
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic_code.config import settings


class BasePersona:
    """Base class for all specialized LLM agent personas."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model_name: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = settings.get_llm(model_name=self.model_name)
        return self._llm

    def invoke_prompt(self, user_content: str) -> str:
        """Invokes the model with system prompt and user input."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]
        response = self.llm.invoke(messages)
        if isinstance(response, AIMessage):
            return str(response.content)
        return str(response)

    @staticmethod
    def extract_json_from_response(text: str) -> Dict[str, Any]:
        """Extracts JSON object from markdown code fences or raw text."""
        # Try ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw JSON block
        raw_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if raw_match:
            try:
                return json.loads(raw_match.group(1))
            except json.JSONDecodeError:
                pass

        return {"raw_text": text}

    @staticmethod
    def extract_code_block(text: str, language: str = "python") -> str:
        """Extracts python/diff code blocks from markdown."""
        pattern = rf"```{language}\s*\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fallback to any generic code block
        generic = re.search(r"```(?:\w+)?\s*\n(.*?)\n```", text, re.DOTALL)
        if generic:
            return generic.group(1).strip()

        return text.strip()
