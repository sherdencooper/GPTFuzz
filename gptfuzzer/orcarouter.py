"""OrcaRouter Cyber Inference provider for GPTFuzzer.

OrcaRouter is an OpenAI-compatible AI gateway for security teams that routes
many models behind one endpoint (``https://api.orcarouter.ai/v1``).  This
module registers a first-class, named provider next to the existing OpenAI,
Claude and Gemini providers:

* credential handling through a small source seam (API key from an explicit
  argument or ``ORCAROUTER_API_KEY``) with masking and terminal ``401``
  classification;
* live model discovery from ``GET /v1/models`` with per-entry capability
  filtering and a small verified cold-start catalog;
* an ``LLM``-compatible adapter used by both the mutation model and the target
  model.

GPTFuzzer has no account-login / OAuth seam, so this provider deliberately
implements API-key authentication only instead of inventing a browser flow.
Only the Python standard library is used, which keeps the provider usable in
the same minimal environment as the rest of the framework.
"""

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROVIDER_ID = "orcarouter"
PROVIDER_NAME = "OrcaRouter Cyber Inference"
DEFAULT_API_BASE = "https://api.orcarouter.ai/v1"
DEFAULT_AUTH_BASE = "https://www.orcarouter.ai"
KEY_ENV_VAR = "ORCAROUTER_API_KEY"
KEY_PREFIX = "sk-orca-"
KEY_DASHBOARD_URL = "https://www.orcarouter.ai/console/api-keys"
CYBER_LANDING_URL = "https://www.orcarouter.ai/cyber"
CYBER_MODELS_URL = "https://www.orcarouter.ai/models?tier=cyber"
DEFAULT_MODEL = "orcarouter/auto"
GENERAL_FALLBACK_MODEL = "orcarouter/fusion"

# Endpoint shapes this client can speak.  OrcaRouter advertises the wire
# protocol(s) a model supports; anything outside this set is not usable by the
# OpenAI-compatible chat path used by GPTFuzzer.
TEXT_ENDPOINT_TYPES = ("openai", "anthropic", "gemini", "openai-response")
NON_TEXT_ENDPOINT_TYPES = (
    "image-generation",
    "openai-video",
    "jina-rerank",
    "embedding",
    "rerank",
)

CATALOG_TIMEOUT_SECONDS = 10.0
CATALOG_MAX_BYTES = 2 * 1024 * 1024
CATALOG_MAX_ITEMS = 500
REQUEST_TIMEOUT_SECONDS = 60.0

# Small, verified cold-start catalog.  It is only used when live discovery
# fails; a successful ``GET /v1/models`` response is authoritative and never
# mixed with these records.
VERIFIED_SEED: Tuple[Dict[str, object], ...] = (
    {
        "id": "openai/gpt-5.5",
        "context_length": 400000,
        "input_modalities": ("text", "image"),
        "reasoning_efforts": ("low", "medium", "high", "xhigh"),
    },
    {
        "id": "anthropic/claude-opus-4.8",
        "context_length": 200000,
        "input_modalities": ("text", "image"),
        "reasoning_efforts": (),
    },
    {
        "id": "google/gemini-3.5-flash",
        "context_length": 1000000,
        "input_modalities": ("text", "image", "audio"),
        "reasoning_efforts": (),
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "context_length": 163840,
        "input_modalities": ("text",),
        "reasoning_efforts": (),
    },
    {
        "id": "orcarouter/auto",
        "context_length": 200000,
        "input_modalities": ("text",),
        "reasoning_efforts": (),
    },
)


class OrcaRouterError(Exception):
    """Base class for provider errors.

    Messages never contain credentials; server payloads are sanitised and
    truncated before being attached.
    """


class MissingCredentialError(OrcaRouterError):
    """No API key was supplied through any supported source."""


class InvalidCredentialError(OrcaRouterError):
    """The supplied API key does not look like an OrcaRouter key."""


class OrcaRouterAuthError(OrcaRouterError):
    """The credential was rejected (401).  Terminal: re-authentication needed."""


class OrcaRouterPermissionError(OrcaRouterError):
    """The credential is valid but not allowed to use this resource (403)."""


class OrcaRouterRateLimitError(OrcaRouterError):
    """The gateway asked the client to slow down (429)."""


class OrcaRouterNetworkError(OrcaRouterError):
    """Transport failure before an HTTP response was received."""


class OrcaRouterCatalogError(OrcaRouterError):
    """The model catalog could not be parsed."""


class UnknownModelError(OrcaRouterError):
    """The requested model is not present in the authoritative live catalog."""


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def mask_key(api_key: Optional[str]) -> str:
    """Return a log-safe representation of an API key."""
    if not api_key:
        return "<unset>"
    if api_key.startswith(KEY_PREFIX) and len(api_key) > len(KEY_PREFIX) + 8:
        return KEY_PREFIX + "...{}".format(api_key[-4:])
    if len(api_key) <= 8:
        return "..."
    return "{}...{}".format(api_key[:4], api_key[-4:])


@dataclass
class CredentialRecord:
    """A resolved credential and its lifecycle state.

    ``generation`` increases whenever the credential is replaced.  A terminal
    ``401`` is applied only to the generation that produced the rejected
    request, so a late failure cannot mark a freshly re-authorised credential
    as broken.
    """

    api_key: str = field(repr=False)
    source: str = "api_key"
    generation: int = 1
    needs_reauth: bool = False

    def rotate(self, api_key: str, source: Optional[str] = None) -> None:
        self.api_key = api_key
        if source is not None:
            self.source = source
        self.generation += 1
        self.needs_reauth = False

    def mark_needs_reauth(self, generation: Optional[int] = None) -> bool:
        """Flag this credential for re-authentication.

        Returns ``False`` when ``generation`` belongs to a superseded
        credential, in which case nothing is changed.
        """
        if generation is not None and generation != self.generation:
            return False
        self.needs_reauth = True
        return True

    def __repr__(self) -> str:  # pragma: no cover - exercised through tests
        return "CredentialRecord(source={!r}, key={}, generation={}, needs_reauth={})".format(
            self.source,
            mask_key(self.api_key),
            self.generation,
            self.needs_reauth,
        )

    __str__ = __repr__


class CredentialSource:
    """Seam between "where the key comes from" and the provider."""

    def resolve(self) -> CredentialRecord:  # pragma: no cover - interface
        raise NotImplementedError


class ApiKeySource(CredentialSource):
    """API-key adapter: explicit argument first, then the environment."""

    def __init__(self, api_key: Optional[str] = None, environ: Optional[Dict[str, str]] = None,
                 env_var: str = KEY_ENV_VAR):
        self.api_key = api_key
        self.environ = environ if environ is not None else os.environ
        self.env_var = env_var

    def resolve(self) -> CredentialRecord:
        key = (self.api_key or self.environ.get(self.env_var) or "").strip()
        if not key:
            raise MissingCredentialError(
                "OrcaRouter API key is required. Pass --orcarouter_key or set "
                "{}. Create a key at {}.".format(self.env_var, KEY_DASHBOARD_URL)
            )
        if not key.startswith(KEY_PREFIX):
            raise InvalidCredentialError(
                "The OrcaRouter API key must start with {!r}. Check the value of "
                "{} or create a new key at {}.".format(KEY_PREFIX, self.env_var, KEY_DASHBOARD_URL)
            )
        return CredentialRecord(api_key=key, source="api_key")


# ---------------------------------------------------------------------------
# Origins
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _validate_origin(url: str, label: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("{} must be an absolute http(s) URL".format(label))
    if parsed.scheme == "http" and parsed.hostname not in (h.strip("[]") for h in _LOOPBACK_HOSTS):
        raise ValueError("{} must use HTTPS unless it points at a loopback host".format(label))
    return url.rstrip("/")


def resolve_api_base(explicit: Optional[str] = None, environ: Optional[Dict[str, str]] = None) -> str:
    """Resolve the inference origin.

    ``ORCA_API_BASE_URL`` is an API base and is used verbatim; ``ORCA_BASE_URL``
    is a shared deployment origin and gets the ``/v1`` API prefix.
    """
    env = environ if environ is not None else os.environ
    if explicit:
        return _validate_origin(explicit, "OrcaRouter API base")
    if env.get("ORCA_API_BASE_URL"):
        return _validate_origin(env["ORCA_API_BASE_URL"], "ORCA_API_BASE_URL")
    shared = env.get("ORCA_BASE_URL")
    if shared:
        return _validate_origin(shared.rstrip("/") + "/v1", "ORCA_BASE_URL")
    return DEFAULT_API_BASE


def resolve_auth_base(explicit: Optional[str] = None, environ: Optional[Dict[str, str]] = None) -> str:
    """Resolve the authentication origin.

    GPTFuzzer only uses API keys, but keeping the two origins explicit makes
    the separation auditable and documents where a future login flow belongs.
    """
    env = environ if environ is not None else os.environ
    if explicit:
        return _validate_origin(explicit, "OrcaRouter auth base")
    if env.get("ORCA_AUTH_BASE_URL"):
        return _validate_origin(env["ORCA_AUTH_BASE_URL"], "ORCA_AUTH_BASE_URL")
    if env.get("ORCA_BASE_URL"):
        return _validate_origin(env["ORCA_BASE_URL"], "ORCA_BASE_URL")
    return DEFAULT_AUTH_BASE


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    id: str
    endpoint_types: Tuple[str, ...]
    input_modalities: Tuple[str, ...]
    context_length: Optional[int] = None
    reasoning_efforts: Tuple[str, ...] = ()
    source: str = "live"

    @property
    def supports_text_chat(self) -> bool:
        return bool(set(self.endpoint_types) & set(TEXT_ENDPOINT_TYPES)) and not (
            set(self.endpoint_types) & set(NON_TEXT_ENDPOINT_TYPES)
        )


@dataclass(frozen=True)
class ModelCatalog:
    models: Tuple[ModelInfo, ...]
    source: str  # "live" | "seed"
    degraded: bool = False
    error: Optional[str] = None

    def ids(self) -> Tuple[str, ...]:
        return tuple(model.id for model in self.models)

    def contains(self, model_id: str) -> bool:
        return any(model.id == model_id for model in self.models)

    def get(self, model_id: str) -> Optional[ModelInfo]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None


def _as_str_tuple(value: object) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def _parse_live_model(item: object) -> Optional[ModelInfo]:
    if not isinstance(item, dict):
        return None
    model_id = str(item.get("id") or "").strip()
    if not model_id:
        return None

    architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
    endpoint_types = _as_str_tuple(
        item.get("supported_endpoint_types") or architecture.get("supported_endpoint_types")
    )
    if not endpoint_types:
        # A record that does not advertise a wire protocol cannot be proven
        # compatible, so it fails closed instead of being guessed at.
        return None

    input_modalities = _as_str_tuple(
        architecture.get("input_modalities") or item.get("input_modalities")
    )

    context_length = item.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        context_length = None

    return ModelInfo(
        id=model_id,
        endpoint_types=endpoint_types,
        input_modalities=input_modalities,
        context_length=context_length,
        reasoning_efforts=_as_str_tuple(item.get("reasoning_efforts")),
        source="live",
    )


def _seed_catalog() -> ModelCatalog:
    models = tuple(
        ModelInfo(
            id=str(record["id"]),
            endpoint_types=TEXT_ENDPOINT_TYPES,
            input_modalities=_as_str_tuple(record.get("input_modalities")),
            context_length=record.get("context_length"),
            reasoning_efforts=_as_str_tuple(record.get("reasoning_efforts")),
            source="seed",
        )
        for record in VERIFIED_SEED
    )
    return ModelCatalog(models=models, source="seed", degraded=True,
                        error="live model catalog unavailable; using verified seed")


def filter_models_for_capability(models: Iterable[ModelInfo], capability: str) -> Tuple[ModelInfo, ...]:
    """Return only models that are provably usable for ``capability``.

    ``chat`` is the only capability GPTFuzzer exposes; unknown capabilities fail
    closed instead of being approximated.
    """
    if capability != "chat":
        raise ValueError("Unsupported OrcaRouter capability {!r}; supported: chat".format(capability))
    return tuple(model for model in models if model.supports_text_chat)


def parse_catalog(payload: bytes, capability: str = "chat") -> Tuple[ModelInfo, ...]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise OrcaRouterCatalogError("model catalog is not valid JSON: {}".format(exc))

    if isinstance(document, dict):
        data = document.get("data")
    else:
        data = document
    if not isinstance(data, list):
        raise OrcaRouterCatalogError("model catalog has no 'data' list")

    parsed: List[ModelInfo] = []
    for item in data[:CATALOG_MAX_ITEMS]:
        model = _parse_live_model(item)
        if model is not None:
            parsed.append(model)
    return filter_models_for_capability(parsed, capability)


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _sanitize(detail: object, limit: int = 200) -> str:
    text = str(detail or "").replace("\n", " ").strip()
    if KEY_PREFIX in text:
        text = text.replace(KEY_PREFIX, KEY_PREFIX + "<redacted>")
    return text[:limit]


class HttpTransport:
    """Minimal injectable HTTP transport."""

    def __call__(self, method: str, url: str, headers: Dict[str, str],
                 body: Optional[bytes], timeout: float) -> Tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read(CATALOG_MAX_BYTES + 1)
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read(CATALOG_MAX_BYTES + 1)
            except Exception:  # pragma: no cover - defensive
                payload = b""
            return exc.code, payload
        except urllib.error.URLError as exc:
            raise OrcaRouterNetworkError("could not reach OrcaRouter: {}".format(_sanitize(exc.reason)))
        except (socket.timeout, TimeoutError) as exc:
            raise OrcaRouterNetworkError("OrcaRouter request timed out: {}".format(_sanitize(exc)))
        except OSError as exc:
            raise OrcaRouterNetworkError("OrcaRouter request failed: {}".format(_sanitize(exc)))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OrcaRouterClient:
    """OpenAI-compatible client scoped to one credential."""

    def __init__(self, credential: CredentialRecord, api_base: Optional[str] = None,
                 transport: Optional[HttpTransport] = None,
                 environ: Optional[Dict[str, str]] = None,
                 request_timeout: float = REQUEST_TIMEOUT_SECONDS):
        self.credential = credential
        self.api_base = resolve_api_base(api_base, environ)
        self.transport = transport or HttpTransport()
        self.request_timeout = request_timeout

    # -- catalog ---------------------------------------------------------
    def discover_models(self, capability: str = "chat",
                        timeout: float = CATALOG_TIMEOUT_SECONDS) -> ModelCatalog:
        """Fetch the authoritative catalog, falling back to the verified seed."""
        query = urllib.parse.urlencode({"capability": capability})
        url = "{}/models?{}".format(self.api_base, query)
        headers = {
            "Authorization": "Bearer {}".format(self.credential.api_key),
            "Accept": "application/json",
            "User-Agent": "gptfuzzer-orcarouter/1",
        }
        try:
            status, body = self.transport("GET", url, headers, None, timeout)
        except OrcaRouterNetworkError as exc:
            logger.warning("%s: %s", PROVIDER_NAME, exc)
            return _seed_catalog()

        if status in (401, 403):
            return self._raise_auth(status, body, capability)
        if status >= 400:
            logger.warning("%s: catalog request failed with HTTP %s", PROVIDER_NAME, status)
            catalog = _seed_catalog()
            return ModelCatalog(models=catalog.models, source="seed", degraded=True,
                                error="catalog request failed with HTTP {}".format(status))
        if len(body) > CATALOG_MAX_BYTES:
            logger.warning("%s: catalog response exceeded the byte bound", PROVIDER_NAME)
            return _seed_catalog()
        try:
            models = parse_catalog(body, capability)
        except OrcaRouterCatalogError as exc:
            logger.warning("%s: %s", PROVIDER_NAME, exc)
            catalog = _seed_catalog()
            return ModelCatalog(models=catalog.models, source="seed", degraded=True,
                                error=str(exc))
        return ModelCatalog(models=models, source="live", degraded=False)

    def _raise_auth(self, status: int, body: bytes, capability: str):
        detail = _sanitize(body)
        if status == 401:
            self.credential.mark_needs_reauth(self.credential.generation)
            raise OrcaRouterAuthError(
                "OrcaRouter rejected the API key (HTTP 401). Create a new key at {} "
                "and pass it with --orcarouter_key or {}.".format(KEY_DASHBOARD_URL, KEY_ENV_VAR)
            )
        raise OrcaRouterPermissionError(
            "OrcaRouter denied access to the {} catalog (HTTP 403). {}".format(capability, detail)
        )

    # -- inference -------------------------------------------------------
    def chat_completion(self, model: str, messages: Sequence[Dict[str, str]],
                        temperature: float = 0.0, max_tokens: int = 512,
                        n: int = 1) -> List[str]:
        url = "{}/chat/completions".format(self.api_base)
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        headers = {
            "Authorization": "Bearer {}".format(self.credential.api_key),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gptfuzzer-orcarouter/1",
        }
        body = json.dumps(payload).encode("utf-8")
        status, response_body = self.transport(
            "POST", url, headers, body, self.request_timeout
        )

        if status == 401:
            self.credential.mark_needs_reauth(self.credential.generation)
            raise OrcaRouterAuthError(
                "OrcaRouter rejected the API key (HTTP 401). Create a new key at {} "
                "and pass it with --orcarouter_key or {}.".format(KEY_DASHBOARD_URL, KEY_ENV_VAR)
            )
        if status == 403:
            raise OrcaRouterPermissionError(
                "OrcaRouter denied model {!r} (HTTP 403). {}".format(model, _sanitize(response_body))
            )
        if status == 429:
            raise OrcaRouterRateLimitError(
                "OrcaRouter rate limit reached (HTTP 429). {}".format(_sanitize(response_body))
            )
        if status >= 400:
            raise OrcaRouterError(
                "OrcaRouter request failed with HTTP {}. {}".format(status, _sanitize(response_body))
            )

        try:
            document = json.loads(response_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise OrcaRouterError("OrcaRouter returned a non-JSON response: {}".format(exc))

        choices = document.get("choices") if isinstance(document, dict) else None
        if not isinstance(choices, list) or not choices:
            raise OrcaRouterError("OrcaRouter response contained no choices")

        results: List[str] = []
        for choice in choices[:n]:
            message = choice.get("message") if isinstance(choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            results.append(content if isinstance(content, str) and content else " ")
        while len(results) < n:
            results.append(" ")
        return results


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


class OrcaRouterLLM:
    """Drop-in replacement for ``OpenAILLM`` backed by OrcaRouter.

    The class mirrors the ``generate`` / ``generate_batch`` contract used by
    GPTFuzzer's mutators and target loop.  In ``gptfuzzer.llm`` it is also
    registered on the ``LLM`` base class.
    """

    def __init__(self,
                 model_path: str = DEFAULT_MODEL,
                 api_key: Optional[str] = None,
                 system_message: Optional[str] = None,
                 api_base: Optional[str] = None,
                 credential_source: Optional[CredentialSource] = None,
                 client: Optional[OrcaRouterClient] = None,
                 catalog: Optional[ModelCatalog] = None,
                 validate_model: bool = True,
                 environ: Optional[Dict[str, str]] = None):
        if client is not None:
            # An injected client already owns a resolved credential; sharing it
            # keeps lifecycle state (needs_reauth) in one place.
            self.client = client
            self.credential = client.credential
        else:
            self.credential = (credential_source or ApiKeySource(api_key, environ)).resolve()
            self.client = OrcaRouterClient(self.credential, api_base, environ=environ)
        self.model_path = model_path or DEFAULT_MODEL
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."
        self.catalog = catalog

        if validate_model:
            self.catalog = catalog or self.client.discover_models("chat")
            self._ensure_supported_model()

    def _ensure_supported_model(self) -> None:
        catalog = self.catalog
        if catalog is None or catalog.source != "live":
            # Without an authoritative catalog the requested id cannot be
            # rejected; the first request will settle it.
            return
        if catalog.contains(self.model_path):
            return
        options = ", ".join(sorted(catalog.ids())[:10]) or "<none>"
        raise UnknownModelError(
            "Model {!r} is not in the live OrcaRouter chat catalog. Choose one of: {}. "
            "Run --list-orcarouter-models to refresh the list.".format(self.model_path, options)
        )

    def refresh_catalog(self) -> ModelCatalog:
        self.catalog = self.client.discover_models("chat")
        self._ensure_supported_model()
        return self.catalog

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10,
                 failure_sleep_time=5):
        """Generate ``n`` responses, mirroring ``OpenAILLM.generate``."""
        if self.credential.needs_reauth:
            logger.error(
                "%s credential needs re-authentication; refusing to call a revoked key. "
                "Create a new key at %s.", PROVIDER_NAME, KEY_DASHBOARD_URL,
            )
            return [" " for _ in range(n)]

        for attempt in range(max_trials):
            try:
                return self.client.chat_completion(
                    self.model_path,
                    self._messages(prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
            except (OrcaRouterAuthError, OrcaRouterPermissionError) as exc:
                # Terminal: there is no refresh grant for a durable API key and
                # a model the key may not use will not become available on retry.
                logger.error("%s: %s", PROVIDER_NAME, exc)
                return [" " for _ in range(n)]
            except OrcaRouterRateLimitError as exc:
                logger.warning("%s: %s Retrying %s / %s...", PROVIDER_NAME, exc, attempt + 1, max_trials)
                time.sleep(failure_sleep_time)
            except OrcaRouterError as exc:
                logger.warning(
                    "%s API call failed due to %s. Retrying %s / %s...",
                    PROVIDER_NAME, exc, attempt + 1, max_trials,
                )
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1,
                       max_trials=10, failure_sleep_time=5):
        results: List[str] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.generate, prompt, temperature, max_tokens, n,
                    max_trials, failure_sleep_time,
                ): prompt
                for prompt in prompts
            }
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

    def __repr__(self) -> str:
        return "{}[model={!r}, key={}]".format(
            PROVIDER_NAME, self.model_path, mask_key(self.credential.api_key)
        )


def list_catalog(api_key: Optional[str] = None, api_base: Optional[str] = None,
                 environ: Optional[Dict[str, str]] = None) -> ModelCatalog:
    """Convenience helper for the CLI listing command."""
    credential = ApiKeySource(api_key, environ).resolve()
    client = OrcaRouterClient(credential, api_base, environ=environ)
    return client.discover_models("chat")
