"""
FastAPI 路由文件
"""
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .model import chat_model

api = APIRouter()


class UploadContent(BaseModel):
    """
    LLM 所需要的参数信息
    """

    chat_history: list[Any]
    top_p: float
    temperature: float


# Routers
@api.put(path="/chat")
async def chat_reply(content: UploadContent):
    """
    获取 ChatGLM3 一次性回复
    """
    return chat_model.chat_reply(content.chat_history, content.top_p, content.temperature)


@api.delete(path="/clear_history")
async def clear_history():
    """
    清除 ChatGLM3 的聊天历史
    """
    chat_model.clear_history()
