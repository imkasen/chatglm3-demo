"""
脏话检测 Demo
"""
import os
import platform

from modelscope import AutoModel, AutoTokenizer, snapshot_download

TOKENIZER = None
MODEL = None
BAD_WORDS: list[str] = ["你好"]
BAD_WORDS_IDS: list | None = None


def init_model():
    """
    初始化 ChatGLM3 模型
    """
    global TOKENIZER, MODEL, BAD_WORDS_IDS  # pylint: disable=W0603

    model_dir: str = snapshot_download("ZhipuAI/chatglm3-6b", revision="master", local_files_only=True)

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if MODEL is None:
        # GPU 推理，默认 FP16 精度加载，需要 13GB 显存
        MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        # MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
        # 模型 4-bit 量化，减少显存压力，6GB 显存即可
        # MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
        # CPU 推理，要求 32GB 内存空间
        # MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float()
        MODEL = MODEL.eval()

    if BAD_WORDS_IDS is None:
        BAD_WORDS_IDS = [TOKENIZER.encode(bad_word, add_special_tokens=False) for bad_word in BAD_WORDS]


def main():
    """
    命令行内对话，流式输出回复
    """
    if (MODEL is None) or (TOKENIZER is None):
        raise RuntimeError("模型未初始化！")

    clear_cmd: str = "cls" if platform.system() == "Windows" else "clear"
    welcome_prompt: str = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

    past_key_values, history = None, []
    print(welcome_prompt)

    while True:
        query: str = input("\n用户：")

        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_cmd)
            print(welcome_prompt)
            continue

        print("\nChatGLM3-6B：", end="")
        current_length: int = 0

        try:
            for response, history, past_key_values in MODEL.stream_chat(
                TOKENIZER,
                query,
                history=history,
                top_p=1,
                temperature=0.01,
                past_key_values=past_key_values,
                return_past_key_values=True,
                bad_words_ids=BAD_WORDS_IDS,
            ):
                # Check if the response contains any bad words
                # if any(bad_word in response for bad_word in BAD_WORDS):
                #     print("我的回答涉嫌了 bad word")
                #     break  # Break the loop if a bad word is detected

                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        except RuntimeError:
            print("生成文本时发生错误，这可能是涉及到设定的敏感词汇。")

        print()


if __name__ == "__main__":
    init_model()
    main()
