"""
API 请求
"""
from typing import Any

import requests


def request_chat_reply(url: str, chat_history: list[Any], top_p: float, temperature: float):
    """
    发送 put 请求来获得 ChatGLM3 回复
    """
    data: dict[str, Any] = {
        "chat_history": chat_history,
        "top_p": top_p,
        "temperature": temperature,
    }
    response = requests.put(url=f"{url}/chat", timeout=60, json=data)
    return response.json()


def clear_history(url: str):
    """
    清除 ChatGLM3 聊天记录
    """
    requests.delete(url=f"{url}/clear_history", timeout=5)
