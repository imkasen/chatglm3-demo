"""
FastAPI 路由文件
"""
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .model import ChatGLM3, ChatGLM3Factory

api = APIRouter()

model_factory = ChatGLM3Factory()


class UploadContent(BaseModel):
    """
    LLM 所需要的参数信息
    """

    chat_history: list[Any]
    top_p: float
    temperature: float


# Routers
@api.post(path="/chat")
async def chat_reply(content: UploadContent, model: ChatGLM3 = Depends(model_factory.get_model)):
    """
    获取 ChatGLM3 单条完整回复
    """
    return model.chat_reply(content.chat_history, content.top_p, content.temperature)


@api.post(path="/stream_chat")
async def stream_chat_reply(content: UploadContent, model: ChatGLM3 = Depends(model_factory.get_model)):
    """
    获取 ChatGLM3 单条流式回复
    """
    reply = model.stream_chat_reply(content.chat_history, content.top_p, content.temperature)
    return StreamingResponse(
        content=reply,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
        media_type="text/event-stream",
    )


@api.delete(path="/clear_history")
async def clear_history(model: ChatGLM3 = Depends(model_factory.get_model)) -> bool:
    """
    清除 ChatGLM3 的聊天历史
    """
    return model.clear_history()
