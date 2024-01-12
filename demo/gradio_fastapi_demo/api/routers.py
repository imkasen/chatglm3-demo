"""
FastAPI 路由文件
"""
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
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
@api.post(path="/chat")
async def chat_reply(content: UploadContent):
    """
    获取 ChatGLM3 单条完整回复
    """
    return chat_model.chat_reply(content.chat_history, content.top_p, content.temperature)


@api.post(path="/stream_chat")
async def stream_chat_reply(content: UploadContent):
    """
    获取 ChatGLM3 单条流式回复
    """
    reply = chat_model.stream_chat_reply(content.chat_history, content.top_p, content.temperature)
    return StreamingResponse(
        content=reply,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@api.delete(path="/clear_history")
async def clear_history() -> bool:
    """
    清除 ChatGLM3 的聊天历史
    """
    return chat_model.clear_history()
