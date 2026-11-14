"""Transport for application-configured inference endpoints."""

import http.client
import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit

from clickhouserag._structured import ModelResponseError, _json_object
from clickhouserag._validation import integer

_MAX_RESPONSE = 16 * 1024 * 1024


class ModelHTTPError(ModelResponseError):
    """An inference endpoint returned a non-success status."""

    def __init__(self, status: int, retry_after: str | None = None) -> None:
        super().__init__(f"Model endpoint returned HTTP {status}")
        self.status = status
        self.retry_after = retry_after


class JsonEndpoint:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: int,
        key: str | None = None,
        key_header: str = "Authorization",
        key_prefix: str = "Bearer ",
        headers: dict[str, str] | None = None,
    ) -> None:
        integer(timeout, "timeout", 3600)
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(ord(char) <= 32 for char in base_url)
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query or fragment"
            )
        if key is not None:
            if not key or any(not 33 <= ord(char) <= 126 for char in key):
                raise ValueError(
                    "api_key must be nonempty and contain no spaces or control characters"
                )
            if parsed.scheme != "https" and not _loopback(parsed.hostname):
                raise ValueError("Use HTTPS for authenticated model endpoints outside loopback")
        self.url = base_url.rstrip("/")
        self._secure = parsed.scheme == "https"
        self._host = parsed.hostname
        self._port = parsed.port
        self._path = parsed.path.rstrip("/")
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        if key is not None:
            self._headers[key_header] = f"{key_prefix}{key}"

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return _json_object(self._send(path, body))

    def post_array(self, path: str, body: dict[str, Any]) -> list[Any]:
        try:
            result = json.loads(self._send(path, body))
        except (ValueError, UnicodeError) as exc:
            raise ModelResponseError("Model did not return valid JSON") from exc
        if not isinstance(result, list):
            raise ModelResponseError("Model response must be a JSON array")
        return result

    def _send(self, path: str, body: dict[str, Any]) -> bytes:
        connection_type = (
            http.client.HTTPSConnection if self._secure else http.client.HTTPConnection
        )
        connection = connection_type(self._host, self._port, timeout=self._timeout)
        try:
            connection.request("POST", self._path + path, json.dumps(body), self._headers)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise ModelHTTPError(response.status, response.getheader("Retry-After"))
            payload = response.read(_MAX_RESPONSE + 1)
            if len(payload) > _MAX_RESPONSE:
                raise ModelResponseError("Model response exceeded the size limit")
            return payload
        except (OSError, http.client.HTTPException) as exc:
            raise ModelResponseError("Model endpoint request failed") from exc
        finally:
            connection.close()


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
