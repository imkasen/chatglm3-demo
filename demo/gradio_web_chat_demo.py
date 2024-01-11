"""
Gradio UI Demo
"""
import os
from pathlib import Path
from typing import Any, LiteralString

import gradio as gr
from modelscope import AutoModel, AutoTokenizer, snapshot_download

TOKENIZER = None
MODEL = None
MESSAGES: list[dict[str, Any]] = []


def init_model():
    """
    初始化 ChatGLM3 模型
    """
    global TOKENIZER, MODEL  # pylint: disable=W0603

    model_path: str = os.path.join(Path().resolve(), "models")

    # Download models: https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary
    model_dir: str = snapshot_download(
        "ZhipuAI/chatglm3-6b",
        revision="master",
        cache_dir=model_path,
        local_files_only=True,
    )
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if MODEL is None:
        MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()


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


def llm_reply(chat_history: list[Any], top_p: float, temperature: float):
    """
    交由 LLM 来处理聊天对话输入并产生结果

    :param chat_history: Gradio 中的聊天历史记录
    :param max_length: LLM 上下文长度
    :param top_p: top p 参数
    :param temperature: temperature 参数
    :yield: 新的聊天历史记录
    """
    global MESSAGES  # pylint: disable=W0603

    if not MESSAGES:  # 构建 LLM 所需要的聊天历史记录格式
        for idx, (user_msg, model_msg) in enumerate(chat_history):
            if idx == len(chat_history) - 1 and not model_msg:
                user_question: str = user_msg  # 用户最新的提问
                break
            if user_msg:
                MESSAGES.append({"role": "user", "content": user_msg})
            if model_msg:
                MESSAGES.append({"role": "assistant", "content": model_msg})
    else:
        user_question = chat_history[-1][0]

    reply, MESSAGES = MODEL.chat(
        TOKENIZER,
        user_question,
        history=MESSAGES,
        top_p=top_p,
        temperature=temperature,
    )
    chat_history[-1][1] = reply
    return chat_history


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


def clear_messages():
    """
    清空聊天历史记录
    """
    global MESSAGES  # pylint: disable=W0603

    MESSAGES = []


# Gradio UI
with gr.Blocks(title="ChatGLM3-6B Gradio Simple Demo") as demo:
    gr.HTML(value="""<h1 align="center">ChatGLM3-6B Gradio Simple Demo</h1>""")
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(placeholder="Input...", lines=5, container=False)
            submit_btn = gr.Button(value="Submit", variant="primary")
        with gr.Column(scale=1):
            empty_btn = gr.ClearButton(value="Clear History", size="sm")

            top_p_input = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.8,
                step=0.01,
                label="Top P",
                interactive=True,
            )
            temperature_input = gr.Slider(
                minimum=0.01,
                maximum=1,
                value=0.6,
                step=0.01,
                label="Temperature",
                interactive=True,
            )

    empty_btn.add(components=[user_input, chatbot])
    empty_btn.click(  # pylint: disable=E1101
        fn=clear_messages,
        inputs=None,
        outputs=None,
    )

    submit_btn.click(  # pylint: disable=E1101
        fn=query_user_input,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
        queue=False,
    ).then(
        fn=llm_reply,
        inputs=[chatbot, top_p_input, temperature_input],
        outputs=chatbot,
    )


if __name__ == "__main__":
    init_model()

    demo.queue().launch(
        inbrowser=True,
        share=False,
        show_api=False,
    )
