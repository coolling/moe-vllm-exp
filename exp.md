## analysis-log-for-predict.py
用于分析每个段内的每一层专家激活频率，验证：同一请求内token的同一层激活专家出现倾斜

## analysis-log-for-route.py
用于分析每次decode的专家激活路径，验证：单一token计算跨不同层的专家激活是很相关的
同时，输出route_experts_distribution.json用作推理时使用

## analysis-log.py
用于分析每层最热门专家
同时，输出rank_experts_distribution.json用作推理时使用

## parquet_data_infer.py
批量执行推理请求，用于实验分析

## rank_expert_analysis.py
对 rank_experts_distribution.json 的实际运用

## route_expert_analysis.py
对 route_experts_distribution.json 的实际运用

# setup.py 
用c编写多线程读取文件绕过python GIL
python setup.py build_ext --inplace



# analysis-acc.py
export HF_ENDPOINT=https://hf-mirror.com
pip install lm-eval
pip install langdetect
pip install immutabledict
pip install unitxt
pip install -e human-eval