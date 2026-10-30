import json

import anthropic
import openai
import requests

from settings import *


class QueryAPIs:
    def __init__(self):
        # Clients are created by set_keys and reused across calls.
        self._claude_client = None
        self._openai_client = None
        self._azure_openai_client = None
        self._gpteal_key = None
        self._gpteal_base_url = None
        # The gateway puts the model in the URL, so one client is built per model.
        self._gpteal_clients = {}

    def set_keys(
        self,
        claude_api_key=None,
        openai_api_key=None,
        azure_openai_api_key=None,
        azure_endpoint=None,
        azure_api_version=AZURE_OPENAI_API_VERSION,
        gpteal_key=None,
        gpteal_base_url=None,
    ):
        """
        Set the API keys.

        azure_openai_api_key and azure_endpoint must be given together;
        azure_endpoint is the resource URL, e.g.
        "https://<resource-name>.openai.azure.com".

        gpteal_key and gpteal_base_url must be given together. GPTeal is an
        Anthropic-compatible gateway reached with
        call_claude(..., provider="gpteal"); gpteal_base_url is the root the
        per-model paths hang off, without a trailing "/anthropic".
        """
        if claude_api_key:
            self._claude_client = anthropic.Anthropic(api_key=claude_api_key)
        if openai_api_key:
            self._openai_client = openai.OpenAI(api_key=openai_api_key)
        if azure_openai_api_key or azure_endpoint:
            if not (azure_openai_api_key and azure_endpoint):
                raise ValueError("Azure OpenAI needs both azure_openai_api_key and azure_endpoint")
            self._azure_openai_client = openai.AzureOpenAI(
                api_key=azure_openai_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        if gpteal_key or gpteal_base_url:
            if not (gpteal_key and gpteal_base_url):
                raise ValueError("GPTeal needs both gpteal_key and gpteal_base_url")
            self._gpteal_key = gpteal_key
            self._gpteal_base_url = gpteal_base_url.rstrip("/")
            self._gpteal_clients = {}

    def _gpteal_client(self, model):
        """Build (and cache) the GPTeal client for one model."""
        if self._gpteal_key is None:
            raise ValueError(
                "No GPTeal client set; call set_keys(gpteal_key=..., gpteal_base_url=...) first"
            )
        if model not in self._gpteal_clients:
            self._gpteal_clients[model] = anthropic.Anthropic(
                base_url=f"{self._gpteal_base_url}/anthropic/{model}",
                # The SDK requires api_key, but the gateway authenticates on the
                # "X-Merck-APIKey" header instead and ignores this value.
                api_key="unused",
                default_headers={"X-Merck-APIKey": self._gpteal_key},
                timeout=REQUEST_TIMEOUT,
            )
        return self._gpteal_clients[model]

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
        provider="anthropic",
        timeout=None,
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
        provider: "anthropic" to call Anthropic directly, or "gpteal" to go
            through the GPTeal gateway. Model names and supported parameters
            differ between the two; GPTeal currently offers
            "claude-opus-4-6-v1", "claude-sonnet-4-6" and
            "claude-haiku-4-5-20251001-v1".
        timeout: per-request timeout in seconds, overriding REQUEST_TIMEOUT.
            Large batched prompts need more than the default.
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
            # newer anthropic SDKs (1.x) no longer take temperature as an argument,
            # so send it in the request body; older models (e.g. Haiku 4.5) still accept it
            kwargs["extra_body"] = {"temperature": temperature}

        if provider == "anthropic":
            if self._claude_client is None:
                raise ValueError("No Claude API key set; call set_keys(claude_api_key=...) first")
            client = self._claude_client
        elif provider == "gpteal":
            client = self._gpteal_client(model)
            # The URL carries the full model slug, but the body wants it without "-v1".
            kwargs["model"] = model.removesuffix("-v1")
        else:
            raise ValueError(f"Unknown provider {provider!r}; expected 'anthropic' or 'gpteal'")

        options = {"max_retries": max_retries}
        if timeout is not None:
            options["timeout"] = timeout
        response = client.with_options(**options).messages.create(**kwargs)

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

    @staticmethod
    def _to_gemini_schema(schema):
        """
        Convert a JSON schema to Gemini's dialect: no "additionalProperties",
        and enums must be strings, so an integer enum becomes a min/max range.
        """
        if isinstance(schema, dict):
            out = {k: QueryAPIs._to_gemini_schema(v) for k, v in schema.items() if k != "additionalProperties"}
            if out.get("type") == "integer" and "enum" in out:
                values = out.pop("enum")
                out["minimum"], out["maximum"] = min(values), max(values)
            return out
        return schema

    def call_gemini(
        self,
        prompt,
        model,
        system=None,
        max_tokens=None,
        temperature=None,
        json_schema=None,
        timeout=None,
    ):
        """
        Send a single-turn prompt to a Gemini model through the GPTeal gateway.

        model: gateway deployment name, e.g. "gemini-2-5-flash" (dashes, not dots).
        max_tokens: cap on output tokens; Gemini counts its hidden thinking tokens too.
        json_schema: forces JSON output. Gemini's schema dialect has no
            "additionalProperties", so that key is dropped.
        """
        if self._gpteal_key is None:
            raise ValueError(
                "No GPTeal client set; call set_keys(gpteal_key=..., gpteal_base_url=...) first"
            )
        # Gemini lives under <gateway>/v2/<model>, next to (not under) the Claude route.
        url = f"{self._gpteal_base_url.rsplit('/', 1)[0]}/v2/{model}"

        generation_config = {}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if json_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = self._to_gemini_schema(json_schema)

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generation_config": generation_config,
        }
        if system is not None:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        response = requests.post(
            url,
            headers={"X-Merck-APIKey": self._gpteal_key},
            json=payload,
            timeout=timeout or REQUEST_TIMEOUT,
        )
        if not response.ok:
            raise requests.HTTPError(f"{response.status_code}: {response.text[:300]}")
        data = response.json()

        # A streamed reply is a list of chunks; a normal one is a single object.
        chunks = data if isinstance(data, list) else [data]
        text = ""
        stop_reason = None
        for chunk in chunks:
            for candidate in chunk.get("candidates", []):
                parts = candidate.get("content", {}).get("parts", [])
                text += "".join(p.get("text", "") for p in parts if not p.get("thought"))
                stop_reason = candidate.get("finishReason") or stop_reason
            if stop_reason is None:
                stop_reason = chunk.get("promptFeedback", {}).get("blockReason")
        return {
            "text": text.strip(),
            "model": chunks[0].get("modelVersion", model),
            "stop_reason": stop_reason,
        }

    def call_openai(
        self,
        prompt,
        model,
        max_retries=MAX_RETRIES,
        system=None,
        max_tokens=None,
        reasoning_effort=None,
        temperature=None,
        json_schema=None,
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
        json_schema: if given, the reply is constrained to JSON matching this
            schema (structured outputs, strict). The JSON is returned as a
            string in "text". Strict mode needs every property listed in
            "required" and "additionalProperties": false.
        """
        if self._openai_client is None:
            raise ValueError("No OpenAI API key set; call set_keys(openai_api_key=...) first")
        return self._call_chat_completions(
            self._openai_client,
            prompt,
            model=model,
            max_retries=max_retries,
            system=system,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            json_schema=json_schema,
        )

    def call_azure_openai(
        self,
        prompt,
        deployment,
        max_retries=MAX_RETRIES,
        system=None,
        max_tokens=None,
        reasoning_effort=None,
        temperature=None,
        json_schema=None,
    ):
        """
        Send a single-turn prompt to an Azure OpenAI deployment.

        Takes the same arguments as call_openai, except that deployment is the
        name of the deployment in the Azure resource rather than a model name.
        """
        if self._azure_openai_client is None:
            raise ValueError(
                "No Azure OpenAI client set; call "
                "set_keys(azure_openai_api_key=..., azure_endpoint=...) first"
            )
        return self._call_chat_completions(
            self._azure_openai_client,
            prompt,
            model=deployment,
            max_retries=max_retries,
            system=system,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            json_schema=json_schema,
        )

    def _call_chat_completions(
        self,
        client,
        prompt,
        model,
        max_retries,
        system,
        max_tokens,
        reasoning_effort,
        temperature,
        json_schema=None,
    ):
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
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }

        response = client.with_options(max_retries=max_retries).chat.completions.create(**kwargs)

        choice = response.choices[0]
        return {
            "text": (choice.message.content or "").strip(),
            "model": response.model,
            "stop_reason": choice.finish_reason,
        }
