"""Tests for the OrcaRouter Cyber Inference provider.

Only the standard library is required: ``python3 -m unittest discover -s tests``.
All credentials in this file are obviously fake.
"""

import json
import unittest
from types import SimpleNamespace

from gptfuzzer import orcarouter
from gptfuzzer.orcarouter import (
    ApiKeySource,
    CredentialRecord,
    InvalidCredentialError,
    MissingCredentialError,
    ModelCatalog,
    ModelInfo,
    OrcaRouterAuthError,
    OrcaRouterClient,
    OrcaRouterError,
    OrcaRouterLLM,
    OrcaRouterNetworkError,
    OrcaRouterPermissionError,
    OrcaRouterRateLimitError,
    UnknownModelError,
    filter_models_for_capability,
    mask_key,
    parse_catalog,
    resolve_api_base,
    resolve_auth_base,
)

FAKE_KEY = "sk-orca-fake-test-0000000000007890"
API_BASE = "https://api.orcarouter.ai/v1"


class FakeTransport:
    """Records calls and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append(SimpleNamespace(
            method=method, url=url, headers=dict(headers), body=body, timeout=timeout,
        ))
        if not self.responses:
            raise AssertionError("unexpected OrcaRouter request: {} {}".format(method, url))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(responses, api_base=API_BASE):
    credential = CredentialRecord(api_key=FAKE_KEY)
    transport = FakeTransport(responses)
    client = OrcaRouterClient(credential, api_base=api_base, transport=transport)
    return client, transport, credential


def catalog_payload(records):
    return json.dumps({"data": records}).encode("utf-8")


def chat_payload(contents):
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": text}} for text in contents]
    }).encode("utf-8")


def live_model(model_id, endpoint_types=("openai",), modalities=("text",), **extra):
    record = {
        "id": model_id,
        "architecture": {
            "input_modalities": list(modalities),
            "supported_endpoint_types": list(endpoint_types),
        },
    }
    record.update(extra)
    return record


class CredentialTests(unittest.TestCase):
    def test_environment_key_is_resolved(self):
        source = ApiKeySource(environ={"ORCAROUTER_API_KEY": FAKE_KEY})
        record = source.resolve()
        self.assertEqual(record.api_key, FAKE_KEY)
        self.assertEqual(record.source, "api_key")
        self.assertFalse(record.needs_reauth)

    def test_explicit_key_wins_over_environment(self):
        source = ApiKeySource("sk-orca-explicit-0000000000", environ={"ORCAROUTER_API_KEY": FAKE_KEY})
        self.assertEqual(source.resolve().api_key, "sk-orca-explicit-0000000000")

    def test_missing_key_raises_actionable_error(self):
        source = ApiKeySource(environ={})
        with self.assertRaises(MissingCredentialError) as ctx:
            source.resolve()
        message = str(ctx.exception)
        self.assertIn("ORCAROUTER_API_KEY", message)
        self.assertIn("https://www.orcarouter.ai/console/api-keys", message)
        self.assertNotIn("sk-orca-", message)

    def test_wrong_prefix_is_rejected_without_echoing_the_value(self):
        source = ApiKeySource(environ={"ORCAROUTER_API_KEY": "sk-wrong-secret-value"})
        with self.assertRaises(InvalidCredentialError) as ctx:
            source.resolve()
        self.assertNotIn("secret-value", str(ctx.exception))

    def test_masking_and_repr_never_expose_the_key(self):
        record = CredentialRecord(api_key=FAKE_KEY)
        self.assertEqual(mask_key(FAKE_KEY), "sk-orca-...7890")
        self.assertNotIn(FAKE_KEY, repr(record))
        self.assertNotIn(FAKE_KEY, str(record))
        self.assertIn("7890", repr(record))

    def test_rotation_advances_generation_and_clears_reauth(self):
        record = CredentialRecord(api_key=FAKE_KEY)
        record.mark_needs_reauth()
        record.rotate("sk-orca-test-rotated-000000000000000000")
        self.assertEqual(record.generation, 2)
        self.assertFalse(record.needs_reauth)

    def test_stale_generation_cannot_mark_a_new_credential(self):
        record = CredentialRecord(api_key=FAKE_KEY)
        stale_generation = record.generation
        record.rotate("sk-orca-test-rotated-000000000000000000")
        self.assertFalse(record.mark_needs_reauth(stale_generation))
        self.assertFalse(record.needs_reauth)
        self.assertTrue(record.mark_needs_reauth(record.generation))
        self.assertTrue(record.needs_reauth)


class OriginTests(unittest.TestCase):
    def test_default_origins_are_separate(self):
        self.assertEqual(resolve_api_base(environ={}), "https://api.orcarouter.ai/v1")
        self.assertEqual(resolve_auth_base(environ={}), "https://www.orcarouter.ai")

    def test_explicit_api_override_is_used_verbatim(self):
        self.assertEqual(
            resolve_api_base("https://gateway.example.test/v1", environ={}),
            "https://gateway.example.test/v1",
        )

    def test_shared_base_gets_the_v1_prefix(self):
        self.assertEqual(
            resolve_api_base(environ={"ORCA_BASE_URL": "https://orca.example.test"}),
            "https://orca.example.test/v1",
        )

    def test_http_is_rejected_for_remote_hosts(self):
        with self.assertRaises(ValueError):
            resolve_api_base("http://api.orcarouter.ai/v1", environ={})

    def test_http_loopback_is_allowed_for_local_deployments(self):
        self.assertEqual(
            resolve_api_base("http://127.0.0.1:8080/v1", environ={}),
            "http://127.0.0.1:8080/v1",
        )

    def test_requests_only_go_to_the_api_origin(self):
        client, transport, _ = make_client([
            (200, catalog_payload([live_model("openai/gpt-5.5")])),
            (200, chat_payload(["ok"])),
        ])
        client.discover_models("chat")
        client.chat_completion("openai/gpt-5.5", [{"role": "user", "content": "hi"}])
        for call in transport.calls:
            self.assertTrue(call.url.startswith(API_BASE), call.url)
            self.assertNotIn("/auth", call.url)
            self.assertNotIn("www.orcarouter.ai", call.url)


class CatalogTests(unittest.TestCase):
    def test_chat_filter_keeps_supported_protocols_and_drops_non_text(self):
        payload = catalog_payload([
            live_model("openai/gpt-5.5"),
            live_model("anthropic/claude-opus-4.8", ("anthropic",)),
            live_model("google/gemini-3.5-flash", ("gemini",), ("text", "image")),
            live_model("openai/gpt-5.5-response", ("openai-response",)),
            live_model("openai/gpt-image-1", ("image-generation",), ("text", "image")),
            live_model("openai/sora-2", ("openai-video",)),
            live_model("jina/reranker-v3", ("jina-rerank",)),
            live_model("vendor/unproven", ()),
            {"id": "vendor/no-architecture"},
            "not-a-record",
        ])
        models = parse_catalog(payload, "chat")
        self.assertEqual(
            [model.id for model in models],
            [
                "openai/gpt-5.5",
                "anthropic/claude-opus-4.8",
                "google/gemini-3.5-flash",
                "openai/gpt-5.5-response",
            ],
        )

    def test_live_discovery_requests_the_chat_capability_with_bearer_auth(self):
        client, transport, _ = make_client([
            (200, catalog_payload([live_model("openai/gpt-5.5")])),
        ])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "live")
        self.assertFalse(catalog.degraded)
        call = transport.calls[0]
        self.assertEqual(call.method, "GET")
        self.assertIn("/v1/models?", call.url)
        self.assertIn("capability=chat", call.url)
        self.assertEqual(call.headers["Authorization"], "Bearer " + FAKE_KEY)
        self.assertNotIn(FAKE_KEY, call.url)

    def test_network_failure_falls_back_to_the_verified_seed(self):
        client, _, _ = make_client([OrcaRouterNetworkError("boom")])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "seed")
        self.assertTrue(catalog.degraded)
        self.assertEqual(len(catalog.models), len(orcarouter.VERIFIED_SEED))

    def test_server_error_falls_back_to_the_verified_seed(self):
        client, _, _ = make_client([(500, b"upstream unavailable")])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "seed")
        self.assertIn("500", catalog.error)

    def test_invalid_json_falls_back_to_the_verified_seed(self):
        client, _, _ = make_client([(200, b"<html>not json</html>")])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "seed")
        self.assertIn("not valid JSON", catalog.error)

    def test_oversized_catalog_is_bounded(self):
        client, _, _ = make_client([(200, b"x" * (orcarouter.CATALOG_MAX_BYTES + 1))])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "seed")

    def test_item_count_is_bounded(self):
        records = [live_model("vendor/model-%d" % index) for index in range(600)]
        client, _, _ = make_client([(200, catalog_payload(records))])
        catalog = client.discover_models("chat")
        self.assertEqual(len(catalog.models), orcarouter.CATALOG_MAX_ITEMS)

    def test_live_success_is_authoritative_and_not_mixed_with_the_seed(self):
        client, _, _ = make_client([
            (200, catalog_payload([live_model("vendor/only-model")])),
        ])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.ids(), ("vendor/only-model",))
        self.assertNotIn("openai/gpt-5.5", catalog.ids())

    def test_empty_live_catalog_is_reported_as_empty_not_seeded(self):
        client, _, _ = make_client([(200, catalog_payload([]))])
        catalog = client.discover_models("chat")
        self.assertEqual(catalog.source, "live")
        self.assertEqual(catalog.models, ())
        self.assertFalse(catalog.degraded)

    def test_unsupported_capability_fails_closed(self):
        with self.assertRaises(ValueError):
            filter_models_for_capability([], "image")

    def test_seed_preserves_verified_metadata(self):
        client, _, _ = make_client([OrcaRouterNetworkError("offline")])
        catalog = client.discover_models("chat")
        gpt55 = catalog.get("openai/gpt-5.5")
        self.assertIsNotNone(gpt55)
        self.assertEqual(gpt55.reasoning_efforts, ("low", "medium", "high", "xhigh"))
        self.assertIn("image", gpt55.input_modalities)
        self.assertEqual(gpt55.source, "seed")

    def test_unknown_endpoint_shapes_fail_closed(self):
        model = orcarouter._parse_live_model({
            "id": "vendor/mystery",
            "architecture": {"input_modalities": ["text"]},
        })
        self.assertIsNone(model)

    def test_gateway_error_on_catalog_marks_the_exact_credential(self):
        client, _, credential = make_client([(401, b'{"error":"revoked"}')])
        with self.assertRaises(OrcaRouterAuthError):
            client.discover_models("chat")
        self.assertTrue(credential.needs_reauth)
        self.assertEqual(credential.generation, 1)


class ChatCompletionTests(unittest.TestCase):
    def test_request_shape_matches_the_openai_wire_format(self):
        client, transport, _ = make_client([(200, chat_payload(["one", "two"]))])
        results = client.chat_completion(
            "orcarouter/auto",
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            temperature=0.25,
            max_tokens=64,
            n=2,
        )
        self.assertEqual(results, ["one", "two"])
        call = transport.calls[0]
        self.assertEqual(call.method, "POST")
        self.assertEqual(call.url, API_BASE + "/chat/completions")
        self.assertEqual(call.headers["Authorization"], "Bearer " + FAKE_KEY)
        body = json.loads(call.body.decode("utf-8"))
        self.assertEqual(body["model"], "orcarouter/auto")
        self.assertEqual(body["temperature"], 0.25)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["n"], 2)
        self.assertEqual(body["messages"][1]["content"], "hi")
        self.assertNotIn(FAKE_KEY, call.body.decode("utf-8"))

    def test_missing_choices_are_padded_with_blank_responses(self):
        client, _, _ = make_client([(200, chat_payload(["only"]))])
        self.assertEqual(
            client.chat_completion("orcarouter/auto", [], n=2),
            ["only", " "],
        )

    def test_401_is_terminal_and_actionable(self):
        client, _, credential = make_client([(401, b'{"error":"revoked"}')])
        with self.assertRaises(OrcaRouterAuthError) as ctx:
            client.chat_completion("orcarouter/auto", [])
        self.assertIn("https://www.orcarouter.ai/console/api-keys", str(ctx.exception))
        self.assertNotIn(FAKE_KEY, str(ctx.exception))
        self.assertTrue(credential.needs_reauth)

    def test_403_and_429_have_distinct_error_types(self):
        client, _, _ = make_client([(403, b'{"error":"denied"}')])
        with self.assertRaises(OrcaRouterPermissionError):
            client.chat_completion("orcacyber-ultra-1.0", [])
        client, _, _ = make_client([(429, b'{"error":"slow down"}')])
        with self.assertRaises(OrcaRouterRateLimitError):
            client.chat_completion("orcarouter/auto", [])

    def test_non_json_and_empty_payloads_are_errors(self):
        client, _, _ = make_client([(200, b"not json")])
        with self.assertRaises(OrcaRouterError):
            client.chat_completion("orcarouter/auto", [])
        client, _, _ = make_client([(200, b'{"choices": []}')])
        with self.assertRaises(OrcaRouterError):
            client.chat_completion("orcarouter/auto", [])


class LLMAdapterTests(unittest.TestCase):
    def build_llm(self, responses, catalog=None, model="orcarouter/auto"):
        client, transport, credential = make_client(responses)
        if catalog is None:
            catalog = ModelCatalog(models=(), source="seed", degraded=True)
        llm = OrcaRouterLLM(
            model_path=model,
            api_key=FAKE_KEY,
            client=client,
            catalog=catalog,
            environ={},
        )
        return llm, transport, credential

    def test_constructor_discovers_the_live_catalog_when_not_injected(self):
        client, transport, _ = make_client([
            (200, catalog_payload([live_model("orcarouter/auto")])),
            (200, chat_payload(["ok"])),
        ])
        llm = OrcaRouterLLM(
            model_path="orcarouter/auto",
            api_key=FAKE_KEY,
            client=client,
            environ={},
        )
        self.assertIsNotNone(llm.catalog)
        self.assertEqual(llm.catalog.source, "live")
        self.assertEqual(llm.generate("seed"), ["ok"])
        self.assertIn("capability=chat", transport.calls[0].url)

    def test_generate_returns_n_responses(self):
        llm, transport, _ = self.build_llm([(200, chat_payload(["mutated"]))])
        self.assertEqual(llm.generate("seed", n=1), ["mutated"])
        self.assertEqual(len(transport.calls), 1)

    def test_401_does_not_retry_and_marks_the_credential(self):
        llm, transport, credential = self.build_llm([(401, b'{"error":"revoked"}')])
        with self.assertLogs("gptfuzzer.orcarouter", level="WARNING"):
            results = llm.generate("seed", n=2, failure_sleep_time=0)
        self.assertEqual(results, [" ", " "])
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(credential.needs_reauth)

    def test_a_revoked_key_is_not_used_again(self):
        llm, transport, credential = self.build_llm([(401, b'{"error":"revoked"}')])
        with self.assertLogs("gptfuzzer.orcarouter", level="ERROR"):
            llm.generate("seed", failure_sleep_time=0)
        credential.mark_needs_reauth()
        calls_before = len(transport.calls)
        with self.assertLogs("gptfuzzer.orcarouter", level="ERROR"):
            self.assertEqual(llm.generate("seed", failure_sleep_time=0), [" "])
        self.assertEqual(len(transport.calls), calls_before)

    def test_403_is_terminal_and_not_retried(self):
        llm, transport, _ = self.build_llm([(403, b'{"error":"model_access_denied"}')])
        with self.assertLogs("gptfuzzer.orcarouter", level="ERROR"):
            results = llm.generate("seed", n=1, failure_sleep_time=0)
        self.assertEqual(results, [" "])
        self.assertEqual(len(transport.calls), 1)

    def test_rate_limit_is_retried(self):
        llm, transport, _ = self.build_llm([
            (429, b'{"error":"slow down"}'),
            (200, chat_payload(["after-wait"])),
        ])
        with self.assertLogs("gptfuzzer.orcarouter", level="WARNING"):
            results = llm.generate("seed", failure_sleep_time=0)
        self.assertEqual(results, ["after-wait"])
        self.assertEqual(len(transport.calls), 2)

    def test_transient_errors_are_retried_then_blank(self):
        llm, transport, _ = self.build_llm([
            (500, b"boom"),
            OrcaRouterNetworkError("connection reset"),
        ])
        with self.assertLogs("gptfuzzer.orcarouter", level="WARNING"):
            results = llm.generate("seed", max_trials=2, failure_sleep_time=0)
        self.assertEqual(results, [" "])
        self.assertEqual(len(transport.calls), 2)

    def test_unknown_model_is_rejected_against_the_live_catalog(self):
        catalog = ModelCatalog(
            models=(ModelInfo("orcarouter/auto", ("openai",), ("text",)),),
            source="live",
        )
        with self.assertRaises(UnknownModelError) as ctx:
            self.build_llm([], catalog=catalog, model="not-a-real-model")
        self.assertIn("orcarouter/auto", str(ctx.exception))

    def test_requested_model_is_allowed_while_the_catalog_is_degraded(self):
        catalog = ModelCatalog(
            models=(ModelInfo("openai/gpt-5.5", ("openai",), ("text",), source="seed"),),
            source="seed",
            degraded=True,
        )
        llm, _, _ = self.build_llm([(200, chat_payload(["ok"]))], catalog=catalog,
                                   model="orcarouter/fusion")
        self.assertEqual(llm.generate("seed"), ["ok"])

    def test_batch_generation_preserves_the_result_count(self):
        llm, _, _ = self.build_llm([
            (200, chat_payload(["a"])),
            (200, chat_payload(["b"])),
        ])
        results = llm.generate_batch(["one", "two"], n=1, failure_sleep_time=0)
        self.assertEqual(sorted(results), ["a", "b"])

    def test_key_is_never_written_to_logs(self):
        llm, _, _ = self.build_llm([(401, b'{"error":"revoked"}')])
        with self.assertLogs("gptfuzzer.orcarouter", level="WARNING") as captured:
            llm.generate("seed", failure_sleep_time=0)
        output = "\n".join(captured.output)
        self.assertNotIn(FAKE_KEY, output)
        self.assertNotIn(FAKE_KEY, repr(llm))

    def test_refresh_catalog_replaces_the_catalog(self):
        llm, _, _ = self.build_llm(
            [(200, catalog_payload([live_model("vendor/fresh-model")]))],
            catalog=ModelCatalog(models=(), source="seed"),
            model="vendor/fresh-model",
        )
        catalog = llm.refresh_catalog()
        self.assertEqual(catalog.ids(), ("vendor/fresh-model",))

    def test_a_selected_model_that_disappears_is_rejected_on_refresh(self):
        llm, _, _ = self.build_llm(
            [(200, catalog_payload([live_model("vendor/other-model")]))],
            catalog=ModelCatalog(
                models=(ModelInfo("vendor/stale-model", ("openai",), ("text",)),),
                source="live",
            ),
            model="vendor/stale-model",
        )
        with self.assertRaises(UnknownModelError):
            llm.refresh_catalog()


class RepositoryWiringTests(unittest.TestCase):
    """Static checks that the provider is registered in the host project."""

    def read(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def test_provider_identity(self):
        self.assertEqual(orcarouter.PROVIDER_NAME, "OrcaRouter Cyber Inference")
        self.assertEqual(orcarouter.DEFAULT_MODEL, "orcarouter/auto")
        self.assertEqual(orcarouter.DEFAULT_API_BASE, "https://api.orcarouter.ai/v1")

    def test_llm_package_exports_the_provider(self):
        self.assertIn("OrcaRouterLLM", self.read("gptfuzzer/llm/__init__.py"))
        source = self.read("gptfuzzer/llm/llm.py")
        self.assertIn("from gptfuzzer.orcarouter import OrcaRouterLLM", source)
        self.assertIn("class OrcaRouterLLM(_OrcaRouterLLM, LLM)", source)

    def test_cli_exposes_the_provider(self):
        source = self.read("gptfuzz.py")
        for expected in ("--orcarouter_key", "--orcarouter_model",
                         "--list-orcarouter-models", "--provider", "--target_provider"):
            self.assertIn(expected, source)

    def test_readme_documents_the_provider(self):
        readme = self.read("README.md")
        self.assertIn("OrcaRouter Cyber Inference", readme)
        self.assertIn("https://api.orcarouter.ai/v1", readme)
        self.assertIn("https://www.orcarouter.ai/cyber", readme)
        self.assertIn("https://www.orcarouter.ai/models?tier=cyber", readme)


if __name__ == "__main__":
    unittest.main()
