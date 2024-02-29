"""
ChatGLM3-6B Model
"""

import threading
from typing import Any

from modelscope import AutoModel, AutoTokenizer, snapshot_download


class ChatGLM3:
    """
    ChatGLM3-6B 对话模型
    """

    def __init__(self, is_quantize: bool = False, is_cpu: bool = False) -> None:
        # TODO: store different histories in db based on different web user requests
        self.history: list[dict[str, Any]] = []

        model_dir: str = snapshot_download("ZhipuAI/chatglm3-6b", revision="master", local_files_only=True)

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

    def format_chat_history(self, chat_history: list[Any]) -> str:
        """
        将 Gradio 聊天记录格式转换为 ChatGLM 聊天记录格式

        :param chat_history: Gradio 格式的聊天记录
        :return: 当前最新的用户提问内容
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
        return user_question

    def chat_reply(self, chat_history: list[Any], top_p: float, temperature: float):
        """
        完整返回模型的单条回复

        :param chat_history: Gradio 格式的聊天记录
        :param top_p: top p 参数
        :param temperature: temperature 参数
        :return: LLM 单条回复
        """
        user_question: str = self.format_chat_history(chat_history)

        reply, self.history = self.model.chat(
            self.tokenizer,
            user_question,
            history=self.history,
            top_p=top_p,
            temperature=temperature,
        )
        return reply

    def stream_chat_reply(self, chat_history: list[Any], top_p: float, temperature: float):
        """
        以流的形式返回模型单条回复

        :param chat_history: Gradio 格式的聊天记录
        :param top_p: top p 参数
        :param temperature: temperature 参数
        :return: LLM 单条回复
        """
        user_question: str = self.format_chat_history(chat_history)

        past_key_values = None
        for reply, self.history, past_key_values in self.model.stream_chat(
            self.tokenizer,
            user_question,
            history=self.history,
            top_p=top_p,
            temperature=temperature,
            past_key_values=past_key_values,
            return_past_key_values=True,
        ):
            # list 不可迭代，在 FastAPI 的 StreamingResponse 中会报错，因此只返回 LLM 回复字符串
            # yield reply + "\n"  # requests: response.iter_lines
            yield reply  # requests: response.iter_content

    def clear_history(self) -> bool:
        """
        清除历史记录
        """
        self.history = []
        return True


class ChatGLM3Factory:
    """
    ChatGLM3 单例模式工厂类
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_instance = ChatGLM3()

    def get_model(self):
        """
        获得 ChatGLM3 模型
        """
        return self.model_instance
