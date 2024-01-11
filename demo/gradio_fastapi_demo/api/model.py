"""
ChatGLM3-6B Model
"""

import os
from pathlib import Path
from typing import Any

from modelscope import AutoModel, AutoTokenizer, snapshot_download


class ChatGLM3:
    """
    ChatGLM3-6B 对话模型
    """

    def __init__(self, is_quantize: bool = False, is_cpu: bool = False) -> None:
        self.history: list[dict[str, Any]] = []

        model_path: str = os.path.join(Path().resolve(), "models")
        # Download models: https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary
        model_dir: str = snapshot_download(
            "ZhipuAI/chatglm3-6b",
            revision="master",
            cache_dir=model_path,
            local_files_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if is_quantize:
            # 模型 4-bit 量化，减少显存压力，6GB 显存即可
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
        elif is_cpu:
            # CPU 推理，要求 32GB 内存空间
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float()
        else:
            # GPU 推理，需要 13GB 显存
            self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        self.model.eval()

    def chat_reply(self, chat_history: list[Any], top_p: float, temperature: float) -> list[Any]:
        """
        一次性返回模型的回复

        :param chat_history: Gradio 格式的聊天记录
        :param top_p: top p 参数
        :param temperature: temperature 参数
        :return: Gradio 格式的新聊天记录
        """
        if not self.history:  # ChatGLM3 格式的聊天记录
            for idx, (user_msg, model_msg) in enumerate(chat_history):
                if idx == len(chat_history) - 1 and not model_msg:
                    user_question: str = user_msg  # 用户最新的提问
                    break
                if user_msg:
                    self.history.append({"role": "user", "content": user_msg})
                if model_msg:
                    self.history.append({"role": "assistant", "content": model_msg})
        else:
            user_question = chat_history[-1][0]

        reply, self.history = self.model.chat(
            self.tokenizer,
            user_question,
            history=self.history,
            top_p=top_p,
            temperature=temperature,
        )

        chat_history[-1][1] = reply
        return chat_history

    def clear_history(self):
        """
        清除历史记录
        """
        self.history = []


# 初始化实例
chat_model = ChatGLM3()
