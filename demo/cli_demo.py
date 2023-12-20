from modelscope import AutoTokenizer, AutoModel, snapshot_download
import platform
import os
from pathlib import Path


TOKENIZER = None
MODEL = None


def init_model():
    """
    初始化 ChatGLM3 模型
    """
    global TOKENIZER, MODEL
    
    MODEL_PATH: str = os.path.join(Path().resolve(), "models")
    
    # Download models: https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary
    model_dir: str = snapshot_download("ZhipuAI/chatglm3-6b", revision="master", cache_dir=MODEL_PATH, local_files_only=True)
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


def main():
    """
    命令行内对话，流式输出回复
    """
    if (MODEL is None) or (TOKENIZER is None):
        raise RuntimeError("模型未初始化！")
    
    CLEAR_CMD: str = 'cls' if platform.system() == 'Windows' else 'clear'
    WELCOME_PROMPT: str = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    
    past_key_values, history = None, []
    print(WELCOME_PROMPT)
    
    while True:
        query: str = input("\n用户：")
        
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(CLEAR_CMD)
            print(WELCOME_PROMPT)
            continue
        
        print("\nChatGLM3-6B：", end="")
        current_length: int = 0
        
        # top_p: 在生成下一个 token 时，从概率分布的前几个最高概率的 token 中进行随机选择的精度阈值。
        # temperature: 控制模型生成文本的随机性和创造性的参数。
        # past_key_values: 一个包含模型过去状态的元组，用于加速生成。
        for response, history, past_key_values in MODEL.stream_chat(TOKENIZER, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            print(response[current_length:], end="", flush=True)
            current_length = len(response)

        print()


if __name__ == "__main__":
    init_model()
    main()
