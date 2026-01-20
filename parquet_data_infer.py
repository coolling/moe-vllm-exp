import polars as pl
# 1. 导入核心类
from vllm import LLM, SamplingParams
# 方法1：读取单个文件
df = pl.read_parquet("/sharenvme/usershome/cyl/test/datasets/dim/sharegpt_short_en_3k/data/train-00000-of-00001-962c97daa9bde1b9.parquet")
# 设置采样参数，控制生成文本的“创造性”
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1024)
if __name__ == '__main__':
    llm = LLM(model="/sharenvme/usershome/cyl/test/model/Isotonic/smol_llama-4x220M-MoE")  # 替换为你的实际模型路径，如 "Qwen/Qwen2.5-7B-Instruct"
    for i in df['conversation']:
        if len(i[0])<1024 and len(i[0])>1:
            prompts = [i[0]]
            # 4. 执行批量推理
            outputs = llm.generate(prompts, sampling_params)








