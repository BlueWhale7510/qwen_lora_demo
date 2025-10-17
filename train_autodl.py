# train_optimized.py
"""
优化版本的训练脚本 - 提高训练效果
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from local_dataset import LocalJsonDataset

def main():
    # 配置参数
    model_name = "./qwen1.5-0.5b-model"
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    print("正在加载预训练模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用FP32避免混合精度问题
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print("预训练模型加载完成！")

    # 增强LoRA配置 - 增加参数数量
    print("配置增强LoRA微调参数...")
    lora_config = LoraConfig(
        r=16,           # 增加秩，提高模型容量
        lora_alpha=32,  # 增加alpha
        lora_dropout=0.1,  # 增加dropout防止过拟合
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],  # 更多目标模块
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    print("加载训练数据集...")
    train_dataset_loader = LocalJsonDataset(
        json_file='finetune_dataset.json',
        tokenizer=tokenizer,
        max_seq_length=512  # 增加序列长度，保留更多上下文
    )
    
    full_dataset = train_dataset_loader.get_dataset()
    dataset = full_dataset.train_test_split(test_size=0.15, seed=42)  # 减少验证集比例
    
    print(f"总样本数: {len(full_dataset)}")
    print(f"训练集: {len(dataset['train'])} 条")
    print(f"验证集: {len(dataset['test'])} 条")
    
    # 数据预处理
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
        )
    
    tokenized_train_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    tokenized_eval_dataset = dataset['test'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['test'].column_names
    )

    # 优化的训练参数
    training_args = TrainingArguments(
        output_dir="./legal_translate_optimized",
        overwrite_output_dir=True,
        # 增加批次大小（利用GPU性能）
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        # 增加训练轮次和学习率
        num_train_epochs=10,           # 增加训练轮次
        learning_rate=2e-4,            # 提高学习率
        # 学习率调度
        lr_scheduler_type="cosine",    # 使用cosine学习率衰减
        warmup_ratio=0.1,              # 10%的warmup
        # 优化日志和保存
        logging_steps=10,
        eval_steps=30,                 # 更频繁的评估
        save_steps=30,                 # 与评估步数匹配
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 禁用混合精度
        fp16=False,
        bf16=False,
        # 数据加载配置
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to=None,
        save_total_limit=3,
        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,             # 权重衰减防止过拟合
    )

    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    print("开始优化训练...")
    print("优化配置:")
    print(f"- 设备: {training_args.device}")
    print(f"- 批次大小: {training_args.per_device_train_batch_size}")
    print(f"- 学习率: {training_args.learning_rate}")
    print(f"- 训练轮次: {training_args.num_train_epochs}")
    print(f"- LoRA秩: 16 (之前: 8)")
    print(f"- 可训练参数: ~1.5M (之前: ~0.8M)")
    print(f"- 序列长度: 512 (之前: 256)")
    
    trainer.train()
    
    print("优化训练完成！保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./legal_translate_optimized")
    print("优化模型保存完成！")

if __name__ == "__main__":
    main()