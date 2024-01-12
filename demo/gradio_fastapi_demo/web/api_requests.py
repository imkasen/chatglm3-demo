"""
API 请求
"""
from typing import Any

import requests


def request_chat_reply(url: str, chat_history: list[Any], top_p: float, temperature: float):
    """
    发送 post 请求来获得 ChatGLM3 的单条完整回复
    """
    data: dict[str, Any] = {
        "chat_history": chat_history,
        "top_p": top_p,
        "temperature": temperature,
    }
    response = requests.post(url=f"{url}/chat", timeout=60, json=data)
    return response.json()


def request_stream_chat_reply(url: str, chat_history: list[Any], top_p: float, temperature: float):
    """
    发送 post 请求来获得 ChatGLM3 的单条流式回复
    """
    data: dict[str, Any] = {
        "chat_history": chat_history,
        "top_p": top_p,
        "temperature": temperature,
    }
    with requests.post(url=f"{url}/stream_chat", timeout=60, json=data, stream=True) as response:
        response.encoding = response.apparent_encoding  # 避免中文内容出现乱码
        # for chunk in response.iter_lines(decode_unicode=True):  # 不可重入
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            yield chunk


def clear_history(url: str):
    """
    清除 ChatGLM3 聊天记录
    """
    response = requests.delete(url=f"{url}/clear_history", timeout=5)
    return response.json()
