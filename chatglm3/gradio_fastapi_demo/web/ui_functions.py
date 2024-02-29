"""
Gradio 组件所需的方法
"""
from typing import Any, LiteralString

import gradio as gr

from .api_requests import clear_history, request_chat_reply, request_stream_chat_reply


def clear_messages(url: str):
    """
    清除 ChatGLM3 历史记录
    """
    if clear_history(url):
        gr.Info("清除聊天历史完成！")


def parse_text(text: str) -> str:
    """
    解析用户输入的文本并转义文本内特殊字符

    :param text: 输入文本
    :return: 处理后的文本
    """
    lines: list[str] = text.split("\n")
    # 移除头尾无意义的空元素
    if lines and lines[0] == "":
        lines = lines[1:]
    if lines and lines[-1] == "":
        lines = lines[:-1]

    in_code_block: bool = False
    for i, line in enumerate(lines):
        if "```" in line:  # 转换 Markdown 代码块
            in_code_block = not in_code_block
            items: list[str] = line.rstrip().split("`")
            if in_code_block:
                lines[i] = f'<pre><code class="language-{items[-1].strip()}">'
            else:
                lines[i] = f"{items[0].strip()}</code></pre>"
        else:
            # 空行
            line: str = line.rstrip()  # 移除空行右侧的空格
            # HTML 转义字符
            line = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;")
                .replace('"', "&quot;")
                .replace("`", r"\`")
                .replace("*", "&ast;")
                .replace("_", "&lowbar;")
                .replace("-", "&#45;")
                .replace(".", "&#46;")
                .replace("!", "&#33;")
                .replace("(", "&#40;")
                .replace(")", "&#41;")
                .replace("$", "&#36;")
                .replace("'", "&#x27;")
                .replace("/", "&#x2F;")
            )
            lines[i] = line + "<br/>" if i + 1 != len(lines) else line
    text = "".join(lines)
    return text


def query_user_input(input_text: str, chat_history: list[Any]) -> tuple[LiteralString, list[Any]]:
    """
    获取用户输入文本并处理

    :param input_text: 用户输入文本
    :param chat_history: 聊天历史记录
    :return: 第一个返回值为空，用于清空输入框内容；第二个返回值往聊天历史记录中插入用户文本
    """
    if input_text != "":
        chat_history += [[parse_text(input_text), None]]  # None 代表不创建回复对话框
    return "", chat_history


def llm_reply(url: str, chat_history: list[Any], top_p: float, temperature: float):
    """
    交由 LLM 来处理聊天对话输入并将单条回复拼接回 Gradio 聊天记录中。

    :param chat_history: Gradio 中的聊天历史记录
    :param top_p: top p 参数
    :param temperature: temperature 参数
    :return: Gradio 格式的新的聊天历史记录
    """
    chat_history[-1][1] = request_chat_reply(url, chat_history, top_p, temperature)
    return chat_history


def llm_stream_reply(url: str, chat_history: list[Any], top_p: float, temperature: float):
    """
    交由 LLM 来处理聊天对话输入并将单条回复拼接回 Gradio 聊天记录中。

    :param chat_history: Gradio 中的聊天历史记录
    :param top_p: top p 参数
    :param temperature: temperature 参数
    :yield: 新的聊天历史记录
    """
    for reply_chunk in request_stream_chat_reply(url, chat_history, top_p, temperature):
        chat_history[-1][1] = reply_chunk
        yield chat_history
