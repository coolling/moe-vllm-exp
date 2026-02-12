import polars as pl
# 1. 导入核心类
from vllm import LLM, SamplingParams
# from vllm.model_executor.tokenizer_utils import get_tokenizer
import os
import time
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "10"
# 方法1：读取单个文件
df = pl.read_parquet("/sharenvme/usershome/cyl/test/datasets/dim/sharegpt_short_en_3k/data/train-00000-of-00001-962c97daa9bde1b9.parquet")
# 设置采样参数，控制生成文本的“创造性”
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1)
# p:30 d:10
if __name__ == '__main__':
    llm = LLM(model="/sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1",enforce_eager=True)  # 替换为你的实际模型路径，如 "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = llm.get_tokenizer()
    c=0
    t=1
    for i in df['conversation']:
        
        if len(i)>=t+1 and len(i[t])<1024 and len(i[t])>1:
            
            prompt=i[t]
            tokens = tokenizer.encode(
                prompt,
                add_special_tokens=True,  # 包含特殊token（如<s>、</s>）
                truncation=False,
                return_tensors=None  # 返回列表而非tensor，更轻量
            )
            token_count = len(tokens)
            print(token_count)
            if token_count>100:
                continue
            c+=1
            print(c)
            if c>30:
                break
            # if c>30:
            #     break
            # print(i[t])
            prompts = [i[t]]
            # 4. 执行批量推理
            outputs = llm.generate(prompts, sampling_params)
   








