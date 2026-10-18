import json

import anthropic
import openai

from settings import *


class QueryAPIs:
    def __init__(self):
        # Clients are created by set_keys and reused across calls.
        self._claude_client = None
        self._openai_client = None

    def set_keys(self, claude_api_key=None, openai_api_key=None):
        """Set the API keys."""
        if claude_api_key:
            self._claude_client = anthropic.Anthropic(api_key=claude_api_key)
        if openai_api_key:
            self._openai_client = openai.OpenAI(api_key=openai_api_key)

    def call_claude(
        self,
        prompt,
        model,
        max_retries=MAX_RETRIES,
        system=None,
        max_tokens=DEFAULT_CLAUDE_MAX_TOKENS,
        effort=None,
        temperature=None,
        json_schema=None,
    ):
        """
        Send a single-turn prompt to a Claude model.

        max_retries: how many times to retry 408/409/429/5xx and connection
            errors (with backoff) before raising.
        max_tokens: cap on output tokens (including thinking). The API requires
            a value; the default is a ceiling, not a target length.
        effort: "low" | "medium" | "high" | "xhigh" | "max". Only supported on
            newer models; leave as None to use the model's default.
        temperature: only accepted by older models (newer ones reject it with
            a 400).
        json_schema: if given, the reply is constrained to JSON matching this
            schema (structured outputs). The JSON is returned as a
            string in "text"; parse it with json.loads.
        """
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system is not None:
            kwargs["system"] = system

        output_config = {}
        if effort is not None:
            output_config["effort"] = effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            kwargs["output_config"] = output_config

        if temperature is not None:
            kwargs["temperature"] = temperature

        if self._claude_client is None:
            raise ValueError("No Claude API key set; call set_keys(claude_api_key=...) first")
        client = self._claude_client.with_options(max_retries=max_retries)
        response = client.messages.create(**kwargs)

        # Newer models return thinking blocks before the answer; keep only the text.
        text = "".join(block.text for block in response.content if block.type == "text")
        return {
            "text": text.strip(),
            "model": response.model,
            "stop_reason": response.stop_reason,
        }

    def call_claude_for_json(self, prompt, model, json_schema, **kwargs):
        """
        Call Claude with a JSON schema and return the parsed JSON as a dict.

        Takes the same arguments as call_claude. Raises ValueError if the
        response was refused or cut off, since it would not be complete JSON.
        """
        result = self.call_claude(prompt, model=model, json_schema=json_schema, **kwargs)
        if result["stop_reason"] != "end_turn":
            raise ValueError(f"stop_reason={result['stop_reason']}")
        return json.loads(result["text"])

    def call_openai(
        self,
        prompt,
        model,
        max_retries=MAX_RETRIES,
        system=None,
        max_tokens=None,
        reasoning_effort=None,
        temperature=None,
    ):
        """
        Send a single-turn prompt to an OpenAI model via Chat Completions, which
        works for both older (gpt-4o) and newer reasoning models.

        max_retries: how many times to retry 408/409/429/5xx and connection
            errors (with backoff) before raising.
        max_tokens: optional cap on output tokens; if None, no limit is sent
            and the model's own maximum applies.
        reasoning_effort: only for reasoning models; leave as None otherwise.
        temperature: not accepted by reasoning models; leave as None for those.
        """
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            kwargs["temperature"] = temperature

        if self._openai_client is None:
            raise ValueError("No OpenAI API key set; call set_keys(openai_api_key=...) first")
        client = self._openai_client.with_options(max_retries=max_retries)
        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "text": (choice.message.content or "").strip(),
            "model": response.model,
            "stop_reason": choice.finish_reason,
        }
