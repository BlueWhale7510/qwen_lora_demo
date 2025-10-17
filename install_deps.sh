#!/bin/bash

echo "AutoDL环境检测..."
echo "PyTorch版本:"
python -c "import torch; print(torch.__version__)"
echo "CUDA可用:"
python -c "import torch; print(torch.cuda.is_available())"

echo "安装必要的依赖..."
pip install transformers==4.37.0
pip install peft==0.8.0
pip install datasets==2.14.0
pip install accelerate==0.25.0

echo "下载模型..."
python -c "
from transformers import AutoTokenizer
model_name = 'Qwen/Qwen1.5-0.5B'
print(f'下载分词器: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('分词器下载完成！')
"

echo "所有依赖安装完成！"