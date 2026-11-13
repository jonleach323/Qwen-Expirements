"""The names and keys a judge's verdicts are kept under (decision 0021).

judge_name names a judge as the scores table and the cache do,
<backend>:<model>. request_key is a verdict's cache key: the SHA-256 of
KEY_VERSION, the backend, the model, the backend's settings that change its
answers (such as the api backend's effort), the rubric and its version, and
the whole request, the text included. text_sha256 is the SHA-256 of a text,
as the tables record what was judged.
"""

import hashlib
import json
from typing import Any

from pushlab.scoring.rubrics import Request

# Raised whenever the key's recipe changes, so no old verdict matches it:
# 2 added the backend's settings.
KEY_VERSION = 2


def text_sha256(text: str) -> str:
    """The SHA-256 of a text's UTF-8 bytes, as hex.

    Example:
        >>> text_sha256("")[:12]
        'e3b0c44298fc'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_name(backend: str, model: str) -> str:
    """The judge as the scores table and the cache name it.

    Example:
        >>> judge_name("api", "claude-haiku-5-5")
        'api:claude-haiku-5-5'
    """
    return f"{backend}:{model}"


def request_key(backend: str, model: str, request: Request, settings: str = "") -> str:
    """The cache key of a request to a backend's model: a SHA-256 of both, of
    the backend's settings, and of everything the request holds, the text
    included."""
    recipe: list[Any] = [
        KEY_VERSION,
        backend,
        model,
        settings,
        request.rubric,
        request.version,
        request.system,
        request.user,
        request.schema,
        request.max_tokens,
    ]
    canonical = json.dumps(recipe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
