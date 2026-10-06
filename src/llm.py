from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# Module-level constants
MIN_INTERVAL_SECONDS: float = 6.5
RETRY_BACKOFFS: list[float] = [5.0, 10.0, 20.0]
DEFAULT_MODEL_NAME: str = "gemini-2.5-flash-lite"
CACHE_DIR: Path = Path(".cache/llm")

# Global timestamp tracker for throttling
_last_call_time: float = 0.0


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM service is unavailable or retries are exhausted."""
    pass


@dataclass
class StructuredResult:
    """Result of generate_structured call."""
    parsed: dict[str, Any]
    llm_calls: int = 1
    model_name: str = ""

    def __getitem__(self, key: str) -> Any:
        if key == "parsed":
            return self.parsed
        if key == "llm_calls":
            return self.llm_calls
        if key == "model_name":
            return self.model_name
        return self.parsed[key]


@dataclass
class ToolLoopResult:
    """Result of run_tool_loop execution."""
    text: str
    tool_calls: int
    tool_names: list[str] = field(default_factory=list)
    llm_calls: int = 0
    model_name: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _throttle(min_interval: float | None = None) -> None:
    """Enforce minimum spacing between SDK calls using time.monotonic."""
    global _last_call_time
    interval = min_interval if min_interval is not None else MIN_INTERVAL_SECONDS
    if interval > 0 and _last_call_time > 0:
        elapsed = time.monotonic() - _last_call_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
    _last_call_time = time.monotonic()


def _is_transient_error(e: Exception) -> bool:
    """Determine whether an error is transient and eligible for retry."""
    if isinstance(e, (errors.APIError, errors.ServerError)):
        code = getattr(e, "code", None)
        if code in (429, 500, 502, 503, 504):
            return True
        msg = str(e).lower()
        if any(w in msg for w in ["429", "503", "resource_exhausted", "unavailable", "rate limit", "quota", "overloaded"]):
            return True
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    msg = str(e).lower()
    if any(w in msg for w in ["429", "503", "resource_exhausted", "service unavailable", "rate limit", "temporarily unavailable", "overloaded"]):
        return True
    return False


def _serialize_for_hash(item: Any) -> Any:
    """Recursively serialize objects for stable JSON hashing."""
    if isinstance(item, (str, int, float, bool)) or item is None:
        return item
    if isinstance(item, (list, tuple)):
        return [_serialize_for_hash(x) for x in item]
    if isinstance(item, dict):
        return {k: _serialize_for_hash(v) for k, v in item.items()}
    if isinstance(item, type) and issubclass(item, BaseModel):
        return item.model_json_schema()
    if isinstance(item, types.Content):
        return {
            "role": item.role,
            "parts": [str(p) for p in (item.parts or [])],
        }
    if isinstance(item, types.Part):
        return str(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return str(item)


def compute_cache_key(
    model: str,
    system: str | None,
    contents: Any,
    tool_declarations: Any,
    schema: Any,
) -> str:
    """Compute sha256 hash of model, system prompt, contents, tool declarations, and schema."""
    payload = {
        "model": model,
        "system": system,
        "contents": _serialize_for_hash(contents),
        "tools": _serialize_for_hash(tool_declarations),
        "schema": _serialize_for_hash(schema),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_disk_cache(cache_key: str, cache_dir: Path | str = CACHE_DIR) -> dict[str, Any] | None:
    """Read cached LLM response from disk if present."""
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cache file %s: %s", cache_path, e)
            return None
    return None


def write_disk_cache(cache_key: str, data: dict[str, Any], cache_dir: Path | str = CACHE_DIR) -> None:
    """Write LLM response to disk cache."""
    try:
        cache_path = Path(cache_dir) / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning("Failed to write cache file %s: %s", cache_key, e)


class _CachedFunctionCall:
    """Mock-compatible function call representation for cached responses."""
    def __init__(self, name: str, args: dict[str, Any]):
        self.name = name
        self.args = args


class _CachedResponse:
    """Mock-compatible response wrapper for cached responses."""
    def __init__(self, text: str | None, function_calls: list[Any] | None = None, parsed: Any = None):
        self.text = text
        self.function_calls = [
            _CachedFunctionCall(fc["name"], fc.get("args", {})) if isinstance(fc, dict) else fc
            for fc in (function_calls or [])
        ] if function_calls is not None else None
        self.parsed = parsed
        self.candidates = []


# NOTE: Gemini 2.5 cannot combine function calling with a response schema in one request,
# so tool use and structured output are separate calls.


class LLMClient:
    """Thin wrapper around google-genai providing throttling, retries, caching, and mock mode."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        use_cache: bool | None = None,
        mock_responder: Callable[..., Any] | None = None,
        client: Any = None,
        cache_dir: Path | str = CACHE_DIR,
    ):
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        self._use_cache = use_cache if use_cache is not None else (os.getenv("LLM_CACHE") == "1")
        self._mock_responder = mock_responder
        self._client = client
        self._cache_dir = Path(cache_dir)

    @property
    def model_name(self) -> str:
        """Active model name."""
        return self._model_name

    def _get_sdk_client(self) -> genai.Client:
        """Lazily initialize genai.Client."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise LLMUnavailableError("GOOGLE_API_KEY is not set and no client was provided.")
        try:
            self._client = genai.Client(api_key=self._api_key)
            return self._client
        except Exception as e:
            raise LLMUnavailableError(f"Failed to initialize Gemini SDK client: {e}") from e

    def _call_generate_content_with_retry(
        self,
        contents: Any,
        config: types.GenerateContentConfig,
        cache_system: str | None = None,
        cache_tools: Any = None,
        cache_schema: Any = None,
        use_cache: bool | None = None,
    ) -> Any:
        """Execute client.models.generate_content with disk caching, throttling, and retries."""
        enable_cache = use_cache if use_cache is not None else self._use_cache

        cache_key = None
        if enable_cache:
            cache_key = compute_cache_key(
                model=self._model_name,
                system=cache_system,
                contents=contents,
                tool_declarations=cache_tools,
                schema=cache_schema,
            )
            cached_data = read_disk_cache(cache_key, cache_dir=self._cache_dir)
            if cached_data is not None:
                return _CachedResponse(
                    text=cached_data.get("text"),
                    function_calls=cached_data.get("function_calls"),
                    parsed=cached_data.get("parsed"),
                )

        backoffs = RETRY_BACKOFFS
        attempts = len(backoffs) + 1
        last_exception: Exception | None = None

        for attempt in range(attempts):
            try:
                _throttle()
                sdk_client = self._get_sdk_client()
                response = sdk_client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )

                if enable_cache and cache_key:
                    fc_list = None
                    if response.function_calls:
                        fc_list = [{"name": fc.name, "args": fc.args} for fc in response.function_calls]
                    
                    parsed_val = None
                    if getattr(response, "parsed", None) is not None:
                        p = response.parsed
                        parsed_val = p.model_dump() if isinstance(p, BaseModel) else p

                    write_disk_cache(
                        cache_key,
                        {
                            "text": response.text,
                            "function_calls": fc_list,
                            "parsed": parsed_val,
                        },
                        cache_dir=self._cache_dir,
                    )

                return response

            except Exception as e:
                last_exception = e
                if attempt < len(backoffs) and _is_transient_error(e):
                    wait_sec = backoffs[attempt]
                    logger.warning("Transient LLM error on attempt %d: %s. Retrying in %ss...", attempt + 1, e, wait_sec)
                    if wait_sec > 0:
                        time.sleep(wait_sec)
                    continue
                logger.error("LLM call failed on attempt %d: %s", attempt + 1, e)
                raise LLMUnavailableError(f"LLM call failed: {e}") from e

        raise LLMUnavailableError(f"LLM call failed after retries: {last_exception}") from last_exception

    def generate_structured(
        self,
        system: str | None,
        user: str,
        schema: type[BaseModel],
        use_cache: bool | None = None,
    ) -> StructuredResult:
        """Generate structured output adhering to a Pydantic schema."""
        # Mock mode check
        if os.getenv("LLM_MOCK") == "1" or self._mock_responder is not None:
            if self._mock_responder is not None:
                res = self._mock_responder(system=system, user=user, schema=schema)
                if isinstance(res, BaseModel):
                    return StructuredResult(parsed=res.model_dump(), llm_calls=1, model_name=self.model_name)
                if isinstance(res, dict):
                    return StructuredResult(parsed=res, llm_calls=1, model_name=self.model_name)
                return StructuredResult(parsed=dict(res), llm_calls=1, model_name=self.model_name)

            # Default canned response based on schema
            dummy_dict = {}
            for field_name, field_info in schema.model_fields.items():
                if field_info.default is not None and field_info.default != PydanticUndefined:
                    dummy_dict[field_name] = field_info.default
                elif field_info.default_factory is not None:
                    dummy_dict[field_name] = field_info.default_factory()
                else:
                    dummy_dict[field_name] = "mock_value"
            return StructuredResult(parsed=dummy_dict, llm_calls=1, model_name=self.model_name)

        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
        )

        response = self._call_generate_content_with_retry(
            contents=user,
            config=config,
            cache_system=system,
            cache_tools=None,
            cache_schema=schema,
            use_cache=use_cache,
        )

        try:
            if getattr(response, "parsed", None) is not None:
                p = response.parsed
                if isinstance(p, BaseModel):
                    parsed_dict = p.model_dump()
                elif isinstance(p, dict):
                    parsed_dict = p
                else:
                    parsed_dict = dict(p)
            else:
                raw_text = response.text or "{}"
                clean_text = raw_text.strip()
                if clean_text.startswith("```"):
                    lines = clean_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    clean_text = "\n".join(lines).strip()
                loaded = json.loads(clean_text)
                validated = schema.model_validate(loaded)
                parsed_dict = validated.model_dump()
            return StructuredResult(parsed=parsed_dict, llm_calls=1, model_name=self.model_name)
        except Exception as e:
            raise LLMUnavailableError(f"Failed to parse structured LLM response: {e}") from e

    def run_tool_loop(
        self,
        system: str | None,
        user: str,
        tool_declarations: list[dict[str, Any]] | None = None,
        tool_impls: dict[str, Callable[..., Any]] | None = None,
        max_iterations: int = 4,
        use_cache: bool | None = None,
    ) -> ToolLoopResult:
        """Run tool calling loop with Gemini function declarations and local tool implementations."""
        if os.getenv("LLM_MOCK") == "1" or self._mock_responder is not None:
            if self._mock_responder is not None:
                res = self._mock_responder(
                    system=system,
                    user=user,
                    tool_declarations=tool_declarations,
                    tool_impls=tool_impls,
                    max_iterations=max_iterations,
                )
                if isinstance(res, ToolLoopResult):
                    return res
                if isinstance(res, dict):
                    return ToolLoopResult(
                        text=res.get("text", "Mock tool loop text"),
                        tool_calls=res.get("tool_calls", 0),
                        tool_names=res.get("tool_names", []),
                        llm_calls=res.get("llm_calls", 1),
                        model_name=self.model_name,
                    )
                return ToolLoopResult(text=str(res), tool_calls=0, tool_names=[], llm_calls=1, model_name=self.model_name)
            return ToolLoopResult(text="Mock tool loop output", tool_calls=0, tool_names=[], llm_calls=1, model_name=self.model_name)

        tools_list = None
        if tool_declarations:
            tools_list = [types.Tool(function_declarations=tool_declarations)]

        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=system,
            tools=tools_list,
        )

        impls = tool_impls or {}
        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=user)])
        ]

        llm_calls = 0
        total_tool_calls = 0
        all_tool_names: list[str] = []
        last_text = ""

        for _ in range(max_iterations):
            response = self._call_generate_content_with_retry(
                contents=contents,
                config=config,
                cache_system=system,
                cache_tools=tool_declarations,
                cache_schema=None,
                use_cache=use_cache,
            )
            llm_calls += 1

            if response.text:
                last_text = response.text

            fcs = response.function_calls
            if not fcs:
                return ToolLoopResult(
                    text=response.text or "",
                    tool_calls=total_tool_calls,
                    tool_names=all_tool_names,
                    llm_calls=llm_calls,
                    model_name=self.model_name,
                )

            # Function calls requested by the model
            model_parts: list[types.Part] = []
            for fc in fcs:
                model_parts.append(types.Part(function_call=types.FunctionCall(name=fc.name, args=fc.args or {})))
            contents.append(types.Content(role="model", parts=model_parts))

            # Execute function calls
            tool_response_parts: list[types.Part] = []
            for fc in fcs:
                fn_name = fc.name
                fn_args = fc.args or {}
                total_tool_calls += 1
                all_tool_names.append(fn_name)

                if fn_name in impls:
                    try:
                        res = impls[fn_name](**fn_args)
                        if not isinstance(res, dict):
                            res = {"result": res}
                    except Exception as e:
                        res = {"status": "unavailable", "error": str(e)}
                else:
                    res = {"status": "unavailable", "error": "unknown tool"}

                tool_response_parts.append(
                    types.Part.from_function_response(name=fn_name, response=res)
                )

            contents.append(types.Content(role="user", parts=tool_response_parts))

        return ToolLoopResult(
            text=last_text,
            tool_calls=total_tool_calls,
            tool_names=all_tool_names,
            llm_calls=llm_calls,
            model_name=self.model_name,
        )


# Module-level client instance and helper functions

_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLMClient."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def generate_structured(
    system: str | None,
    user: str,
    schema: type[BaseModel],
    use_cache: bool | None = None,
    client: LLMClient | None = None,
) -> StructuredResult:
    """Module-level wrapper for generate_structured."""
    cli = client or get_llm_client()
    return cli.generate_structured(system=system, user=user, schema=schema, use_cache=use_cache)


def run_tool_loop(
    system: str | None,
    user: str,
    tool_declarations: list[dict[str, Any]] | None = None,
    tool_impls: dict[str, Callable[..., Any]] | None = None,
    max_iterations: int = 4,
    use_cache: bool | None = None,
    client: LLMClient | None = None,
) -> ToolLoopResult:
    """Module-level wrapper for run_tool_loop."""
    cli = client or get_llm_client()
    return cli.run_tool_loop(
        system=system,
        user=user,
        tool_declarations=tool_declarations,
        tool_impls=tool_impls,
        max_iterations=max_iterations,
        use_cache=use_cache,
    )
