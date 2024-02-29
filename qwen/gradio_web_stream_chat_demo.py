"""
Qwen Gradio Demo
"""

import gc
from typing import Any

import gradio as gr
import mdtex2html
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, snapshot_download

DEFAULT_CKPT_PATH = "Qwen/Qwen-7B-Chat-Int4"
TOKENIZER = None
MODEL = None
CONFIG = None

# pylint: disable=C0116, E1101, W0603


def load_model():
    global TOKENIZER, MODEL, CONFIG

    # model_dir: str = snapshot_download(DEFAULT_CKPT_PATH, revision="master", local_files_only=True)
    model_dir: str = snapshot_download(DEFAULT_CKPT_PATH, revision="master")

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            resume_download=True,
        )

    if MODEL is None:
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            resume_download=True,
        ).eval()

    if CONFIG is None:
        CONFIG = GenerationConfig.from_pretrained(
            model_dir,
            trust_remote_code=True,
            resume_download=True,
        )


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text: str):
    lines: list[str] = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items: list[str] = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line: str = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(_query: str, _chatbot: list[tuple[Any, Any]], _task_history: list[tuple[Any, Any]]):
    _chatbot.append((_parse_text(_query), ""))
    full_response: str = ""

    for response in MODEL.chat_stream(TOKENIZER, _query, history=_task_history, generation_config=CONFIG):
        _chatbot[-1] = (_parse_text(_query), _parse_text(response))

        yield _chatbot
        full_response = _parse_text(response)

    _task_history.append((_query, full_response))


def regenerate(_chatbot: list[tuple[Any, Any]], _task_history: list[tuple[Any, Any]]):
    if not _task_history:
        yield _chatbot
        return
    item = _task_history.pop(-1)
    _chatbot.pop(-1)
    yield from predict(item[0], _chatbot, _task_history)


def reset_user_input():
    return gr.update(value="")


def reset_state(_chatbot: list[tuple[Any, Any]], _task_history: list[tuple[Any, Any]]):
    _task_history.clear()
    _chatbot.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _chatbot


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>Qwen-Chat Bot</center>""")

    chatbot = gr.Chatbot(label="Qwen-Chat", elem_classes="control-height")
    query = gr.Textbox(lines=2, label="Input")
    task_history = gr.State([])

    with gr.Row():
        empty_btn = gr.Button("🧹 Clear History (清除历史)")
        submit_btn = gr.Button("🚀 Submit (发送)")
        regen_btn = gr.Button("🤔️ Regenerate (重试)")

    submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
    submit_btn.click(reset_user_input, [], [query])
    empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
    regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

    gr.Markdown(
        """\
<font size=2>Note: This demo is governed by the original license of Qwen. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)"""
    )


if __name__ == "__main__":
    load_model()
    demo.queue().launch(
        inbrowser=True,
        share=False,
        show_api=False,
    )
