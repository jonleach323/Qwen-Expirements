"""The judge's backends by name, as the commands choose them (decision 0021).

make_backend gives the backend a command asks for: "api", Claude through
the Anthropic API (pushlab.scoring.api_backend), asking claude-haiku-5-5
unless told another model; or "local", an open model that a vLLM server
serves (pushlab.scoring.local_backend), which has no default: the command
names the model the server serves. Importing this module imports neither
the anthropic package, which an ApiBackend imports only when made, nor
anything heavy, so that every --help answers at once.
"""

from pushlab.scoring.api_backend import DEFAULT_API_MODEL, ApiBackend
from pushlab.scoring.judge import Backend
from pushlab.scoring.local_backend import LOCAL_URL, NEEDS_MODEL, LocalBackend

BACKENDS = ("api", "local")
# Each backend's model when none is given; "local" has none.
DEFAULT_MODELS = {"api": DEFAULT_API_MODEL}


def judge_model(backend: str, model: str | None = None) -> str:
    """The model a backend's judge asks: model, when given, else the
    backend's default. ValueError for an unknown backend, and for "local"
    with no model, as it has no default.

    Example:
        >>> judge_model("api"), judge_model("local", "granite-4.1-8b")
        ('claude-haiku-5-5', 'granite-4.1-8b')
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown judge backend {backend!r}; there are {', '.join(BACKENDS)}"
        )
    if model is not None and model.strip():
        return model.strip()
    if backend in DEFAULT_MODELS:
        return DEFAULT_MODELS[backend]
    raise ValueError(NEEDS_MODEL)


def make_backend(
    backend: str, model: str | None = None, base_url: str | None = None
) -> Backend:
    """The backend named, asking judge_model(backend, model): an ApiBackend,
    or a LocalBackend at base_url, else LOCAL_URL.

    ValueError for a base_url with "api", which calls the Anthropic API, and
    as judge_model, ApiBackend and LocalBackend refuse: an unknown backend,
    "local" with no model, no anthropic package or no ANTHROPIC_API_KEY, a
    URL that is not http:// or https:// to a host.

    Example:
        >>> backend = make_backend("local", "granite-4.1-8b")
        >>> backend.name, backend.model
        ('local', 'granite-4.1-8b')
    """
    name = judge_model(backend, model)
    if backend == "api":
        if base_url is not None:
            raise ValueError(
                "the api backend calls the Anthropic API, so it takes no server "
                "URL: a URL is for the local backend"
            )
        return ApiBackend(name)
    return LocalBackend(name, LOCAL_URL if base_url is None else base_url)
