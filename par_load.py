import threading
import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
import re
import os
import gc
import tempfile
import fcntl
import struct
import torch
from safetensors.torch import load_file
from safetensors import safe_open
import torch
import json
import numpy as np


def parallel_read_and_parse(file_path, weight_name, max_workers=None, chunk_size_mb=8):
    """并行读取并解析safetensors文件"""
    # 第一步：先读取文件头（单独线程）
    print("读取safetensors文件头...")
    start=time.time()
    # safetensors文件格式：前8字节是header长度，然后是json header，最后是tensor数据
    with open(file_path, "rb") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            # 读取header长度（8字节，little-endian）
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise ValueError("文件太小或格式错误")
            
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            
            # 读取header JSON
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError("header长度不匹配")
            header = json.loads(header_bytes.decode('utf-8'))          
            # 获取文件总大小和tensor数据位置
            data_start_pos = 8 + header_len  # 数据开始位置
   
            tensor_info = None
            for key, info in header.items():
                if key == weight_name:
                    tensor_info = info
                    break
            # 获取tensor的数据范围
            dtype = tensor_info["dtype"]
            shape = tensor_info["shape"]
            data_offsets = tensor_info["data_offsets"]
            tensor_start = data_start_pos + data_offsets[0]
            tensor_end = data_start_pos + data_offsets[1]
            tensor_size = tensor_end - tensor_start
            
            print(f"目标tensor: {weight_name}")
            print(f"  位置: {tensor_start}-{tensor_end} (大小: {tensor_size} 字节)")
            print(f"  形状: {shape}, 类型: {dtype}")
            
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    elapsed_ms = (time.time() - start) * 1000
    
    print(f"[DEBUG] 读取safetensors文件头 {elapsed_ms:.2f} ms")
    # 第二步：并行读取tensor数据
    start=time.time()
    print(f"并行读取tensor数据 ({tensor_size/1024/1024:.2f} MB)...")
    
    # 计算需要的线程数和块大小
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = min(8, cpu_count * 2)
    
    # 计算每个线程读取的块
    chunk_size = chunk_size_mb * 1024 * 1024
    num_chunks = math.ceil(tensor_size / chunk_size)
    
    # 限制线程数不超过块数
    actual_workers = min(max_workers, num_chunks)
    
    def read_file_chunk_to_memory(file_path, start_pos, chunk_size, worker_id):
        """读取文件的指定块到内存"""
        try:
            with open(file_path, "rb") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # 共享锁，允许多个读取
                try:
                    f.seek(start_pos)
                    data = f.read(chunk_size)
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            return worker_id, data, len(data), None
        except Exception as e:
            return worker_id, None, 0, str(e)
    # 创建线程池
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        
        # 提交所有数据块读取任务
        for i in range(num_chunks):
            chunk_start = tensor_start + i * chunk_size
            actual_chunk_size = min(chunk_size, tensor_end - chunk_start)
            
            future = executor.submit(
                read_file_chunk_to_memory,
                file_path,
                chunk_start,
                actual_chunk_size,
                i
            )
            futures.append(future)
        # 收集所有数据块
        chunks = [None] * num_chunks
        errors = []
        for future in as_completed(futures):
            worker_id, data, size, error = future.result()
            
            if error:
                errors.append(f"块 {worker_id}: {error}")
            elif data:
                chunks[worker_id] = data
           

        
        if errors:
            print(f"读取错误: {errors}")
            return None, []
    
    # 第三步：合并数据并创建tensor
    print("合并数据并创建tensor...")
    elapsed_ms = (time.time() - start) * 1000
    print(f"[DEBUG] 读取数据 {elapsed_ms:.2f} ms")
    start=time.time()
    # 合并所有数据块
    # combined_data=bytearray(tensor_size)
    
    offset = 0
    return chunks
    
    
def load_expert_weight_par(
    hf_weights_files: list[str],
    weight_name: str = "",
    expert_dim: int = 256,
    parallel_loading: bool = True,
    max_workers: int = None,
    chunk_size_mb: int = 4
):
    """加载专家权重（支持并行读取文件到内存）
    
    每个线程读取文件的不同部分到内存，然后在内存中合并
    """
    
    # 查找目标文件
    target_file = None
    st_file = hf_weights_files[0]
    st_file_dir = os.path.dirname(st_file)
    experts_dir = os.path.join(st_file_dir, "experts")
    safe_name = weight_name
    possible_file = os.path.join(experts_dir, f"{safe_name}.safetensors")
    
    if os.path.exists(possible_file):
        target_file = possible_file
        print(f"找到目标文件: {target_file}")

    if target_file is None:
        print(f"权重 {weight_name} 未找到")
        return None

    try:
        start=time.time()
        chunks = parallel_read_and_parse(
                target_file,
                weight_name,
                max_workers=max_workers,
                chunk_size_mb=chunk_size_mb
            )
      
        elapsed_ms = (time.time() - start) * 1000
        print(f"[DEBUG] parallel_read_and_parse {elapsed_ms:.2f} ms")

        return chunks
        
    except Exception as e:
        print(f"加载失败: {e}")
        
        return None
import time
start=time.time()
chunks=load_expert_weight_par(["/sharenvme/usershome/cyl/test/model/mistralai/Mixtral-8x7B-Instruct-v0.1/model-00001-of-00019.safetensors"],"model.layers.0.block_sparse_moe.experts.0.w1.weight")
elapsed_ms = (time.time() - start) * 1000
# print(t)
print(f"[DEBUG] all {elapsed_ms:.2f} ms")

combined_data=bytearray(14336*4096*2)
print(f"bytearray 大小: {len(combined_data)} 字节")
def parse_to_tensor(combined_data,chunks):
    start=time.time()
    offset=0
    mv = memoryview(combined_data)
    for chunk in chunks:
        mv[offset:offset + len(chunk)] = chunk
        offset += len(chunk)

    elapsed_ms = (time.time() - start) * 1000
    print(f"[DEBUG] 合并所有数据块 {elapsed_ms:.2f} ms")

    start=time.time()
    uint16_data = np.frombuffer(combined_data, dtype=np.uint16)
    # 使用torch的bfloat16支持
    tensor = torch.frombuffer(combined_data, dtype=torch.bfloat16).reshape([14336, 4096])


    elapsed_ms = (time.time() - start) * 1000
    print(f"[DEBUG] 转为tensor {elapsed_ms:.2f} ms")
    return tensor
    
t=parse_to_tensor(combined_data,chunks)
print(t)