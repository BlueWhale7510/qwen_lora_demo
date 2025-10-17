"""
本地JSON数据集加载器
专门用于法律术语翻译任务的数据处理
"""

import json
from datasets import Dataset


class LocalJsonDataset:
    """
    本地JSON数据集加载器类
    专门处理法律术语翻译任务的instruction-input-output格式数据
    """
    
    def __init__(self, json_file, tokenizer, max_seq_length=1024):
        """
        初始化数据集加载器
        
        参数:
        - json_file: JSON数据文件路径
        - tokenizer: 分词器对象，用于处理文本
        - max_seq_length: 最大序列长度，默认1024
        """
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """
        加载并处理JSON数据文件
        
        返回:
        - Dataset对象: 处理后的Hugging Face数据集
        """
        # 读取JSON文件内容
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []  # 存储处理后的文本样本
        
        print(f"检测到instruction-input-output格式数据，共{len(data)}条样本")
        
        # 处理法律术语翻译数据格式
        for item in data:
            # 构建格式为：指令\n输入\n输出 的训练样本
            instruction = item['instruction']
            input_text = item['input']
            output_text = item['output']
            
            # 构建训练文本格式
            text = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n{output_text}"
            texts.append(text)

        # 创建数据集字典，添加'text'字段以适配SFTTrainer
        dataset_dict = {
            'text': texts  # SFTTrainer要求的字段名
        }
        
        # 转换为Hugging Face Dataset对象
        dataset = Dataset.from_dict(dataset_dict)
        print(f"数据集加载完成，共{len(dataset)}条样本")
        return dataset

    def get_dataset(self):
        """
        获取处理后的数据集
        
        返回:
        - Dataset对象: 处理后的Hugging Face数据集
        """
        return self.dataset

    def tokenize_dataset(self):
        """
        对数据集进行tokenize处理
        
        返回:
        - Dataset对象: tokenize后的数据集
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
            )
        
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        return tokenized_dataset