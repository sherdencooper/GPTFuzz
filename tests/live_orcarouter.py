"""Live connectivity check for the OrcaRouter Cyber Inference provider.

Runs against the real gateway using ``ORCAROUTER_API_KEY``.  It exercises the
code path shipped in this pull request:

1. ``GET https://api.orcarouter.ai/v1/models?capability=chat`` through
   :class:`gptfuzzer.orcarouter.OrcaRouterClient`;
2. a real chat completion through :class:`gptfuzzer.orcarouter.OrcaRouterLLM`.

The key is never printed.  Exit code 0 means a real completion was received.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptfuzzer.orcarouter import (  # noqa: E402
    KEY_ENV_VAR,
    KEY_DASHBOARD_URL,
    ModelCatalog,
    OrcaRouterAuthError,
    OrcaRouterClient,
    OrcaRouterError,
    OrcaRouterLLM,
    PROVIDER_NAME,
    ApiKeySource,
)

# Cyber models are tried first when the key has the cyber tier unlocked.  The
# public model is the documented fallback for keys that do not.
CANDIDATES = (
    ("orca/orcacyber-ultra-1.0", "cyber"),
    ("orca/orcacyber-zero-1.5", "cyber"),
    ("deepseek/deepseek-v4.1-flash", "public"),
)


def main() -> int:
    api_key = os.environ.get(KEY_ENV_VAR, "").strip()
    if not api_key:
        print("FAIL: {} is not set; live verification requires a real key".format(KEY_ENV_VAR))
        return 1

    try:
        credential = ApiKeySource(api_key).resolve()
    except OrcaRouterError as exc:
        print("FAIL: {}".format(exc))
        return 1
    client = OrcaRouterClient(credential)

    catalog = None
    try:
        catalog = client.discover_models("chat")
        print("catalog: source={} degraded={} chat_models={}".format(
            catalog.source, catalog.degraded, len(catalog.models)))
        for model in sorted(catalog.models, key=lambda item: item.id)[:10]:
            print("  - {}".format(model.id))
    except OrcaRouterAuthError as exc:
        print("FAIL: model catalog rejected the key: {}".format(exc))
        return 1
    except OrcaRouterError as exc:
        print("catalog unavailable (degraded): {}".format(exc))

    catalog_ids = set(catalog.ids()) if catalog is not None and catalog.source == "live" else set()
    ordered = []
    if os.environ.get("ORCAROUTER_LIVE_MODEL"):
        ordered.append(os.environ["ORCAROUTER_LIVE_MODEL"])
    for model_id, _kind in CANDIDATES:
        ordered.append(model_id)
    if catalog_ids:
        # Prefer models the live catalog actually advertises to this key.
        ordered = [model_id for model_id in ordered if model_id in catalog_ids] or ordered

    seen = set()
    for model_id in ordered:
        if model_id in seen:
            continue
        seen.add(model_id)
        # Membership was already checked against the live catalog above; use the
        # seed catalog here so a model that is visible in the console but not in
        # the list response can still be attempted once.
        llm = OrcaRouterLLM(
            model_path=model_id,
            client=client,
            catalog=ModelCatalog(models=(), source="seed", degraded=True),
            system_message="You are a helpful assistant.",
        )
        try:
            results = llm.generate(
                "Reply with the single word: pong",
                temperature=0.0,
                max_tokens=16,
                n=1,
                max_trials=1,
                failure_sleep_time=0,
            )
        except OrcaRouterError as exc:
            print("model {} failed: {}".format(model_id, exc))
            if credential.needs_reauth:
                print("FAIL: the key was rejected; create a new one at {}".format(KEY_DASHBOARD_URL))
                return 1
            continue

        text = (results[0] if results else "").strip()
        if text:
            print("OK: {} answered via {} (chars={})".format(PROVIDER_NAME, model_id, len(text)))
            preview = " ".join(text.split())[:80]
            print("response_preview: {}".format(preview))
            return 0
        if credential.needs_reauth:
            print("FAIL: the key was rejected; create a new one at {}".format(KEY_DASHBOARD_URL))
            return 1
        print("model {} returned an empty response (likely 403/429 for this key)".format(model_id))

    print("FAIL: no OrcaRouter model produced a completion")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
