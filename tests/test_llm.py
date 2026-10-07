from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from google.genai import errors, types
from pydantic import BaseModel

import src.llm as llm
from src.llm import (
    LLMClient,
    LLMUnavailableError,
    StructuredResult,
    ToolLoopResult,
    compute_cache_key,
    generate_structured,
    run_tool_loop,
)


class SampleSchema(BaseModel):
    summary: str
    approved: bool
    cost: float


class TestLLM(unittest.TestCase):
    def setUp(self):
        self.orig_min_interval = llm.MIN_INTERVAL_SECONDS
        self.orig_retry_backoffs = llm.RETRY_BACKOFFS
        self.orig_last_call_time = llm._last_call_time
        self.orig_default_client = llm._default_client
        # Tests set MIN_INTERVAL_SECONDS to 0 by default
        llm.MIN_INTERVAL_SECONDS = 0
        llm.RETRY_BACKOFFS = [0.0, 0.0, 0.0]
        llm._last_call_time = 0.0
        llm._default_client = None

        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def tearDown(self):
        llm.MIN_INTERVAL_SECONDS = self.orig_min_interval
        llm.RETRY_BACKOFFS = self.orig_retry_backoffs
        llm._last_call_time = self.orig_last_call_time
        llm._default_client = self.orig_default_client

    def test_throttle_spacing(self):
        """Throttle enforces spacing using time.monotonic and time.sleep."""
        llm.MIN_INTERVAL_SECONDS = 6.5
        llm._last_call_time = 100.0

        with patch("time.monotonic", side_effect=[102.0, 102.0, 106.5]), patch("time.sleep") as mock_sleep:
            llm._throttle()
            mock_sleep.assert_called_once_with(4.5)

    def test_retry_then_success(self):
        """Transient errors trigger retries and succeed if a later attempt succeeds."""
        mock_sdk = MagicMock()
        success_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text='{"summary": "ok", "approved": true, "cost": 10.0}')],
                    )
                )
            ]
        )
        # Fail twice with 429 then succeed
        mock_sdk.models.generate_content.side_effect = [
            errors.APIError(code=429, response_json={"message": "Resource exhausted"}),
            errors.APIError(code=503, response_json={"message": "Service unavailable"}),
            success_response,
        ]

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        result = client.generate_structured("system", "user", SampleSchema)

        self.assertEqual(mock_sdk.models.generate_content.call_count, 3)
        self.assertEqual(result.parsed["summary"], "ok")
        self.assertTrue(result.parsed["approved"])
        self.assertEqual(result.parsed["cost"], 10.0)

    def test_retry_exhausted_raises_llm_unavailable_error(self):
        """Exhausting all retries raises LLMUnavailableError."""
        mock_sdk = MagicMock()
        mock_sdk.models.generate_content.side_effect = errors.APIError(
            code=429, response_json={"message": "Rate limit exceeded"}
        )

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        with self.assertRaises(LLMUnavailableError):
            client.generate_structured("system", "user", SampleSchema)

        # Initial attempt + 3 retries = 4 attempts total
        self.assertEqual(mock_sdk.models.generate_content.call_count, 4)

    def test_cache_hit_makes_no_second_sdk_call(self):
        """When cache is enabled, identical requests hit the disk cache with no 2nd SDK call."""
        mock_sdk = MagicMock()
        success_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text='{"summary": "cached", "approved": true, "cost": 50.0}')],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.return_value = success_response

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=True)
        res1 = client.generate_structured("system prompt", "user query", SampleSchema)
        self.assertEqual(mock_sdk.models.generate_content.call_count, 1)
        self.assertEqual(res1.parsed["summary"], "cached")

        # Second call with same inputs
        res2 = client.generate_structured("system prompt", "user query", SampleSchema)
        self.assertEqual(mock_sdk.models.generate_content.call_count, 1)
        self.assertEqual(res2.parsed["summary"], "cached")

    def test_cache_off_by_default(self):
        """Cache is disabled by default, causing repeated requests to make multiple SDK calls."""
        mock_sdk = MagicMock()
        success_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text='{"summary": "fresh", "approved": false, "cost": 0.0}')],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.return_value = success_response

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLM_CACHE", None)
            client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir)
            self.assertFalse(client._use_cache)

            client.generate_structured("sys", "usr", SampleSchema)
            client.generate_structured("sys", "usr", SampleSchema)
            self.assertEqual(mock_sdk.models.generate_content.call_count, 2)

    def test_mock_mode_makes_no_sdk_call(self):
        """Mock mode (via env or mock_responder) makes no SDK call."""
        mock_sdk = MagicMock()

        # Via mock_responder
        custom_responder = lambda system, user, schema: {"summary": "custom", "approved": True, "cost": 1.0}
        client = LLMClient(api_key="fake-key", client=mock_sdk, mock_responder=custom_responder)
        res = client.generate_structured("sys", "usr", SampleSchema)
        self.assertEqual(mock_sdk.models.generate_content.call_count, 0)
        self.assertEqual(res.parsed["summary"], "custom")

        # Via LLM_MOCK env var
        with patch.dict(os.environ, {"LLM_MOCK": "1"}):
            client_env_mock = LLMClient(api_key="fake-key", client=mock_sdk)
            res_env = client_env_mock.generate_structured("sys", "usr", SampleSchema)
            self.assertEqual(mock_sdk.models.generate_content.call_count, 0)
            self.assertIn("summary", res_env.parsed)

            tool_res = client_env_mock.run_tool_loop("sys", "usr")
            self.assertEqual(mock_sdk.models.generate_content.call_count, 0)
            self.assertIsInstance(tool_res, ToolLoopResult)

    def test_run_tool_loop_parallel_tools_and_counts(self):
        """run_tool_loop executes multiple parallel tool calls in one turn, records counts, and returns final text."""
        mock_sdk = MagicMock()

        # Turn 1: Model requests two parallel tool calls
        turn1_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(function_call=types.FunctionCall(name="get_budget", args={"dept": "Eng"})),
                            types.Part(function_call=types.FunctionCall(name="check_risk", args={"vendor": "Acme"})),
                        ],
                    )
                )
            ]
        )

        # Turn 2: Model receives tool outputs and provides final answer
        turn2_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Budget is ok and risk is verified.")],
                    )
                )
            ]
        )

        mock_sdk.models.generate_content.side_effect = [turn1_response, turn2_response]

        budget_impl = MagicMock(return_value={"status": "ok", "available": 50000})
        risk_impl = MagicMock(return_value={"status": "ok", "risk_level": "low"})

        tool_declarations = [
            {"name": "get_budget", "description": "Get department budget"},
            {"name": "check_risk", "description": "Check vendor risk"},
        ]
        tool_impls = {
            "get_budget": budget_impl,
            "check_risk": risk_impl,
        }

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        result = client.run_tool_loop("sys prompt", "user query", tool_declarations, tool_impls)

        budget_impl.assert_called_once_with(dept="Eng")
        risk_impl.assert_called_once_with(vendor="Acme")

        self.assertEqual(result.text, "Budget is ok and risk is verified.")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.tool_names, ["get_budget", "check_risk"])
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.model_name, client.model_name)

    def test_run_tool_loop_max_iterations(self):
        """run_tool_loop terminates when max_iterations is reached."""
        mock_sdk = MagicMock()
        infinite_tool_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(function_call=types.FunctionCall(name="ping", args={})),
                        ],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.return_value = infinite_tool_response

        ping_impl = MagicMock(return_value={"pong": True})
        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        result = client.run_tool_loop(
            system="sys",
            user="user",
            tool_declarations=[{"name": "ping"}],
            tool_impls={"ping": ping_impl},
            max_iterations=3,
        )

        self.assertEqual(result.llm_calls, 3)
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(ping_impl.call_count, 3)

    def test_run_tool_loop_unknown_tool_handled(self):
        """run_tool_loop handles unknown tool names gracefully without raising exceptions."""
        mock_sdk = MagicMock()
        turn1_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(function_call=types.FunctionCall(name="unknown_tool_xyz", args={})),
                        ],
                    )
                )
            ]
        )
        turn2_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Handled unknown tool.")],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.side_effect = [turn1_response, turn2_response]

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        result = client.run_tool_loop("sys", "user", tool_declarations=[], tool_impls={})

        self.assertEqual(result.text, "Handled unknown tool.")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.tool_names, ["unknown_tool_xyz"])
        self.assertEqual(result.llm_calls, 2)

    def test_generate_structured_pydantic_schema(self):
        """generate_structured returns parsed dictionary conforming to Pydantic schema."""
        mock_sdk = MagicMock()
        json_text = '{"summary": "Purchase Figma licenses", "approved": true, "cost": 4500.0}'
        success_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=json_text)],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.return_value = success_response

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        res = client.generate_structured("system prompt", "request info", SampleSchema)

        self.assertIsInstance(res, StructuredResult)
        self.assertEqual(res.parsed["summary"], "Purchase Figma licenses")
        self.assertTrue(res.parsed["approved"])
        self.assertEqual(res.parsed["cost"], 4500.0)
        self.assertEqual(res.llm_calls, 1)

    def test_key_never_appears_in_logs(self):
        """Secret API key never appears in captured log output."""
        secret_key = "AIzaSySecretApiKeyMustNeverAppearInLogs12345"
        mock_sdk = MagicMock()
        mock_sdk.models.generate_content.side_effect = errors.APIError(
            code=429, response_json={"message": "Transient rate limit"}
        )

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        target_logger = logging.getLogger("src.llm")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)

        try:
            client = LLMClient(api_key=secret_key, client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
            try:
                client.generate_structured("sys", "user", SampleSchema)
            except LLMUnavailableError:
                pass

            logs_text = log_stream.getvalue()
            self.assertNotIn(secret_key, logs_text)
        finally:
            target_logger.removeHandler(handler)

    def test_run_tool_loop_cache_hit(self):
        """With use_cache=True, the first turn (user prompt) is cached. Function-call turns
        are NOT cached to preserve thought_signatures. The final text-only turn is cached."""
        mock_sdk = MagicMock()
        turn1_resp = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(name="get_data", args={"k": "v"}))],
                    )
                )
            ]
        )
        turn2_resp = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Final answer from tool data.")],
                    )
                )
            ]
        )
        # First run: turn 1 (user prompt, cacheable) + turn 2 (after tool, not cached) = 2 SDK calls
        mock_sdk.models.generate_content.side_effect = [turn1_resp, turn2_resp]

        tool_impls = {"get_data": lambda k: {"result": "ok"}}
        tool_decls = [{"name": "get_data", "description": "Get data"}]

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=True)
        res1 = client.run_tool_loop("sys", "user query", tool_decls, tool_impls)
        self.assertEqual(mock_sdk.models.generate_content.call_count, 2)
        self.assertEqual(res1.text, "Final answer from tool data.")
        self.assertEqual(res1.tool_calls, 1)
        self.assertEqual(res1.tool_names, ["get_data"])
        self.assertEqual(res1.llm_calls, 2)

        # Second run: turn 1 hits cache (0 SDK calls), but turn 2 (after tool
        # execution with new contents) needs a fresh SDK call = 1 more SDK call
        mock_sdk.models.generate_content.side_effect = [turn2_resp]
        res2 = client.run_tool_loop("sys", "user query", tool_decls, tool_impls)
        self.assertEqual(mock_sdk.models.generate_content.call_count, 3)  # 2 + 1
        self.assertEqual(res2.text, res1.text)
        self.assertEqual(res2.tool_calls, res1.tool_calls)
        self.assertEqual(res2.tool_names, res1.tool_names)
        self.assertEqual(res2.llm_calls, res1.llm_calls)

    def test_run_tool_loop_function_response_role_user(self):
        """Assert that contents passed to generate_content on turn 2 end with a Content whose role == 'user' and whose part is a function_response."""
        mock_sdk = MagicMock()
        turn1_resp = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(function_call=types.FunctionCall(name="check_status", args={}))],
                    )
                )
            ]
        )
        turn2_resp = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Done.")],
                    )
                )
            ]
        )
        mock_sdk.models.generate_content.side_effect = [turn1_resp, turn2_resp]

        tool_impls = {"check_status": lambda: {"status": "all_good"}}
        tool_decls = [{"name": "check_status", "description": "Check status"}]

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        client.run_tool_loop("sys", "user msg", tool_decls, tool_impls)

        self.assertEqual(mock_sdk.models.generate_content.call_count, 2)
        turn2_call_args = mock_sdk.models.generate_content.call_args_list[1]
        contents_turn2 = turn2_call_args.kwargs["contents"]

        last_content = contents_turn2[-1]
        self.assertEqual(last_content.role, "user")
        self.assertTrue(len(last_content.parts) > 0)
        self.assertIsNotNone(last_content.parts[0].function_response)
        self.assertEqual(last_content.parts[0].function_response.name, "check_status")

    def test_thought_signature_preserved_in_turn2(self):
        """When the SDK response Part carries thought_signature bytes, the Content
        passed to generate_content on turn 2 must contain that same Part with the
        signature intact (not a rebuilt FunctionCall part)."""
        mock_sdk = MagicMock()

        # Build a Part with a FunctionCall that carries a thought_signature
        sig_bytes = b"\x01\x02\x03\x04secret-signature-bytes"
        fc_part = types.Part(
            function_call=types.FunctionCall(
                name="lookup_vendor",
                args={"name": "Acme"},
            ),
            thought_signature=sig_bytes,
        )
        turn1_content = types.Content(role="model", parts=[fc_part])
        turn1_response = types.GenerateContentResponse(
            candidates=[types.Candidate(content=turn1_content)]
        )

        turn2_response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text="Vendor Acme found.")],
                    )
                )
            ]
        )

        mock_sdk.models.generate_content.side_effect = [turn1_response, turn2_response]

        tool_impls = {"lookup_vendor": lambda name: {"status": "ok", "risk": "low"}}
        tool_decls = [{"name": "lookup_vendor", "description": "Look up vendor"}]

        client = LLMClient(api_key="fake-key", client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
        result = client.run_tool_loop("sys", "Find vendor Acme", tool_decls, tool_impls)

        self.assertEqual(result.text, "Vendor Acme found.")
        self.assertEqual(mock_sdk.models.generate_content.call_count, 2)

        # Inspect the contents passed to turn 2
        turn2_call_args = mock_sdk.models.generate_content.call_args_list[1]
        contents_turn2 = turn2_call_args.kwargs["contents"]

        # contents_turn2 should be: [user, model (with sig), user (tool response)]
        self.assertEqual(len(contents_turn2), 3)

        model_content = contents_turn2[1]
        self.assertEqual(model_content.role, "model")
        self.assertEqual(len(model_content.parts), 1)

        # The model part should be the ORIGINAL Part object, not a rebuilt one
        preserved_part = model_content.parts[0]
        self.assertIsNotNone(preserved_part.function_call)
        self.assertEqual(preserved_part.function_call.name, "lookup_vendor")
        # The thought_signature must be preserved
        self.assertEqual(preserved_part.thought_signature, sig_bytes)
        # It should be the exact same Part object
        self.assertIs(preserved_part, fc_part)

    def test_tool_loop_error_logged_without_api_key(self):
        """When a tool-loop LLM call raises, a warning is logged with error text
        but the API key is never included."""
        secret_key = "AIzaSyNeverLogThis99999"
        mock_sdk = MagicMock()
        mock_sdk.models.generate_content.side_effect = RuntimeError(
            f"Bad request with key {secret_key}"
        )

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        target_logger = logging.getLogger("src.llm")
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.DEBUG)

        try:
            client = LLMClient(api_key=secret_key, client=mock_sdk, cache_dir=self.test_dir, use_cache=False)
            with self.assertRaises(LLMUnavailableError):
                client.run_tool_loop("sys", "user", tool_declarations=[{"name": "t"}])

            logs_text = log_stream.getvalue()
            self.assertIn("Tool-loop LLM call failed", logs_text)
            self.assertNotIn(secret_key, logs_text)
            self.assertIn("[REDACTED]", logs_text)
        finally:
            target_logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
