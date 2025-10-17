"""
法律术语翻译模型评估脚本
对比基础模型（Zero-shot）与微调模型（LoRA）的效果
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_models():
    """加载基础模型和微调模型"""
    base_model_name = "./qwen1.5-0.5b-model"
    
    print("=" * 60)
    print("加载模型中...")
    print("=" * 60)
    
    # 加载基础模型
    print("1. 加载基础模型...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        padding_side="left"
    )
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()
    print("✅ 基础模型加载完成")
    
    # 加载微调模型
    print("2. 加载微调模型...")
    finetuned_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ),
        "./legal_translate_optimized"
    )
    finetuned_model.eval()
    print("✅ 微调模型加载完成")
    
    return base_model, base_tokenizer, finetuned_model, base_tokenizer

def generate_baseline_answer(model, tokenizer, legal_text):
    """
    基础模型生成 - 使用优化的Zero-shot Prompt
    """
    # 优化的Zero-shot Prompt
    prompt = f"""请将以下法律术语用普通人能听懂的大白话解释清楚，要求：
1. 准确反映原意
2. 使用日常口语表达
3. 避免专业术语

法律术语：{legal_text}

通俗解释："""
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            top_p=0.9
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取prompt之后的内容
    if prompt in decoded:
        return decoded[len(prompt):].strip()
    return decoded.strip()

def generate_finetuned_answer(model, tokenizer, legal_text):
    """
    微调模型生成 - 使用训练时的格式
    """
    input_text = f"### 指令:\n请将以下法律术语解释成普通人能听懂的大白话。\n\n### 输入:\n{legal_text}\n\n### 输出:\n"
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            top_p=0.9
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if input_text in decoded:
        return decoded[len(input_text):].strip()
    return decoded.strip()

def load_test_data():
    """从训练数据中提取5条作为测试集"""
    with open('finetune_dataset.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # 选择5条作为测试集（选择不同法律领域的样本）
    test_indices = [0, 10, 20, 30, 40]  # 均匀分布选择
    test_data = [all_data[i] for i in test_indices if i < len(all_data)]
    
    print(f"从数据集中选取了 {len(test_data)} 条测试样本")
    return test_data

def main():
    # 加载模型
    base_model, base_tokenizer, finetuned_model, finetuned_tokenizer = load_models()
    
    # 加载测试数据
    test_data = load_test_data()
    
    print("\n" + "=" * 80)
    print("法律术语翻译模型对比评估")
    print("=" * 80)
    
    results = []
    
    for i, sample in enumerate(test_data, 1):
        legal_text = sample['input']
        reference = sample['output']
        
        print(f"\n📋 测试案例 {i}:")
        print(f"【法律术语】: {legal_text}")
        print(f"【参考输出】: {reference}")
        
        # 基础模型生成
        print(f"\n🔵 【基础模型 - Zero-shot】:")
        baseline_output = generate_baseline_answer(base_model, base_tokenizer, legal_text)
        print(f"   {baseline_output}")
        
        # 微调模型生成
        print(f"\n🟢 【微调模型 - LoRA】:")
        finetuned_output = generate_finetuned_answer(finetuned_model, finetuned_tokenizer, legal_text)
        print(f"   {finetuned_output}")
        
        # 保存结果
        result = {
            "sample_id": i,
            "legal_text": legal_text,
            "reference": reference,
            "baseline_output": baseline_output,
            "finetuned_output": finetuned_output
        }
        results.append(result)
        
        print("-" * 80)
    
    # 保存详细结果
    with open("baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成简洁的对比报告
    generate_comparison_report(results)
    
    print(f"\n✅ 评估完成！")
    print(f"📊 详细结果保存至: baseline_comparison.json")
    print(f"📋 对比报告保存至: comparison_report.txt")

def generate_comparison_report(results):
    """生成简洁的对比报告"""
    report = ["法律术语翻译模型对比评估报告", "=" * 50, ""]
    
    for result in results:
        report.append(f"样本 {result['sample_id']}:")
        report.append(f"法律术语: {result['legal_text']}")
        report.append(f"参考输出: {result['reference']}")
        report.append("")
        report.append("基础模型 (Zero-shot):")
        report.append(f"  {result['baseline_output']}")
        report.append("")
        report.append("微调模型 (LoRA):")
        report.append(f"  {result['finetuned_output']}")
        report.append("-" * 60)
        report.append("")
    
    report.append("评估说明:")
    report.append("1. 忠实原文 - 解释是否准确反映法律含义")
    report.append("2. 通俗易懂 - 是否使用普通人能理解的语言") 
    report.append("3. 表达自然 - 语言是否流畅自然")
    report.append("")
    report.append("请从以上三个维度对两种模型的输出进行评分（1-5分）")
    
    with open("comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    # 同时在控制台输出简洁版本
    print("\n" + "=" * 60)
    print("简洁对比结果:")
    print("=" * 60)
    
    for result in results:
        print(f"\n案例 {result['sample_id']}: {result['legal_text']}")
        print(f"参考: {result['reference']}")
        print(f"基础: {result['baseline_output'][:80]}...")
        print(f"微调: {result['finetuned_output'][:80]}...")

if __name__ == "__main__":
    main()