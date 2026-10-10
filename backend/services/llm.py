"""Alibaba Cloud Bailian (Model Studio) through its OpenAI-compatible chat API, using plain HTTP.

Docs: https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
The API key must come from the same region as the address, or the service answers 401.
"""
from __future__ import annotations

import requests

REGIONS = {
    # region: (address without a workspace, address with one)
    "cn": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "https://{ws}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
    "intl": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "https://{ws}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
    "us": ("https://dashscope-us.aliyuncs.com/compatible-mode/v1", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"),
}
DEFAULT_MODEL = "qwen-plus"
TIMEOUT_S = 120


class LLMError(RuntimeError):
    pass


def base_url(region: str, workspace: str = "") -> str:
    plain, with_ws = REGIONS.get(region, REGIONS["cn"])
    ws = (workspace or "").strip()
    return with_ws.format(ws=ws) if ws else plain


class BailianClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, region: str = "cn", workspace: str = "", session=None,
                 temperature: float = 0.3):
        if not api_key:
            raise LLMError("还没有填写百炼 API key / no Bailian API key")
        self._key, self.model, self._url = api_key, model or DEFAULT_MODEL, base_url(region, workspace) + "/chat/completions"
        self._session = session or requests.Session()
        self._temperature = temperature

    def chat(self, messages: list[dict]) -> dict:
        """Send a chat and return the reply text with token usage."""
        try:
            resp = self._session.post(self._url, headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                                      json={"model": self.model, "messages": messages, "temperature": self._temperature},
                                      timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            raise LLMError(f"连接百炼失败，请检查网络 / could not reach Bailian: {type(exc).__name__}") from exc
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code in (401, 403):
            raise LLMError("百炼拒绝了这个 API key：请确认 key 正确，并且和所选地域一致 / Bailian rejected the API key; check it matches the region")
        if resp.status_code == 404:
            raise LLMError(f"百炼找不到模型 {self.model}，请在设置里换一个模型名 / model not found")
        if resp.status_code == 429:
            raise LLMError("百炼请求太频繁或额度不足，请稍后再试 / rate limited or out of quota")
        if resp.status_code >= 400:
            msg = ((body.get("error") or {}).get("message") if isinstance(body, dict) else None) or f"HTTP {resp.status_code}"
            raise LLMError(f"百炼返回错误：{msg} / Bailian error")
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("百炼的回复格式无法识别 / unexpected reply")
        usage = body.get("usage") or {}
        return {"text": text, "prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": int(usage.get("completion_tokens") or 0)}
