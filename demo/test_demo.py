from modelscope import AutoTokenizer, AutoModel, snapshot_download
import time
import os
from pathlib import Path


PROJ_ROOT: Path = Path().resolve()
MODEL_PATH: str = os.path.join(PROJ_ROOT, "models")
TOKENIZER = None
MODEL = None


def init_model():
    global TOKENIZER, MODEL
    
    # https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary
    model_dir: str = snapshot_download("ZhipuAI/chatglm3-6b", revision = "master", cache_dir=MODEL_PATH, local_files_only=False)
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if MODEL is None:
        # GPU 推理，默认 FP16 精度加载，需要 13GB 显存
        # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
        # 模型 4-bit 量化，减少显存压力
        MODEL = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
        # CPU 推理，要求 32GB 内存空间
        # model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float()
        MODEL = MODEL.eval()


def chat(content: str, history):
    global TOKENIZER, MODEL
    
    if (MODEL is None) or (TOKENIZER is None):
        raise RuntimeError("模型未初始化！")
    
    t: float = time.perf_counter()
    print(f"问：{content}")
    response, history = MODEL.chat(TOKENIZER, content, history=history)
    print(f"答：{response}")
    print(f"花费时间: {(time.perf_counter() - t):.2f}s\n")
    return history


if __name__ == "__main__":
    init_model()
    
    history = chat("你好！", [])
    history = chat("圣诞节是什么时候？", history)
