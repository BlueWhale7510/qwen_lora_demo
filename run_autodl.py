"""
法律术语翻译模型推理脚本 - 优化版本
加载优化微调后的LoRA模型进行法律术语翻译
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

def main():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name()}")
    
    # 配置参数 - 修改模型路径
    base_model_name = "./qwen1.5-0.5b-model"
    lora_model_path = "./legal_translate_optimized"  # 修改为优化版本的路径
    
    # 检查微调模型是否存在
    if os.path.exists(lora_model_path):
        print("检测到优化微调模型，正在加载...")
        try:
            # 首先加载基础模型
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, 
                trust_remote_code=True,
                padding_side="left"
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 然后加载LoRA权重
            model = PeftModel.from_pretrained(
                base_model,
                lora_model_path,
            )
            print("✅ 优化微调模型加载成功！")
            print(f"模型路径: {lora_model_path}")
            
        except Exception as e:
            print(f"加载优化微调模型失败: {e}")
            print("尝试加载原始版本微调模型...")
            # 尝试加载原始版本
            try:
                model = PeftModel.from_pretrained(
                    base_model,
                    "./legal_translate_model",  # 原始版本路径
                )
                print("✅ 原始微调模型加载成功！")
            except:
                print("回退到原始模型...")
                tokenizer = AutoTokenizer.from_pretrained(
                    base_model_name, 
                    trust_remote_code=True,
                    padding_side="left"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print("原始模型加载成功！")
    else:
        print("未找到优化微调模型，检查其他版本...")
        # 检查其他可能的模型路径
        possible_paths = [
            "./legal_translate_optimized",
            "./legal_translate_model", 
            "./legal_translate_aggressive"
        ]
        
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到模型: {path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name, 
                        trust_remote_code=True,
                        padding_side="left"
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(base_model, path)
                    print(f"✅ {path} 加载成功！")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"❌ {path} 加载失败: {e}")
        
        if not model_loaded:
            print("加载原始模型...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, 
                trust_remote_code=True,
                padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print("原始模型加载成功！")

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 将模型切换到推理模式
    model.eval()
    print("模型准备完成！")

    def generate_answer(legal_text):
        """生成法律术语的白话解释"""
        # 构建输入提示 - 与训练时的格式保持一致
        input_text = f"### 指令:\n请将以下法律术语解释成普通人能听懂的大白话。\n\n### 输入:\n{legal_text}\n\n### 输出:\n"
        
        # 分词
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # 与训练时保持一致
            padding=True
        )
        
        # 移动到模型所在的设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                early_stopping=True
            )
        
        # 解码输出
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取输出部分（在输入文本之后的内容）
        if input_text in decoded_output:
            response = decoded_output[len(input_text):].strip()
        else:
            response = decoded_output.strip()
            
        return response

    # 测试推理
    print("\n" + "="*50)
    print("测试推理 - 优化版本")
    print("="*50)
    
    test_texts = [
        "原告应承担举证责任，否则将面临败诉风险。",
        "本合同的不可抗力条款旨在规避缔约双方在无法预见、无法避免且无法克服的客观情况下的违约责任。",
        "担保人对主债务承担连带保证责任。"
    ]
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\n测试案例 {i}:")
        print(f"输入: {test_text}")
        try:
            result = generate_answer(test_text)
            print(f"输出: {result}")
        except Exception as e:
            print(f"生成失败: {e}")
        print("-" * 50)

    # 交互式推理
    print("\n" + "="*50)
    print("进入交互式翻译模式！")
    print("请输入法律术语，输入'exit'退出:")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['exit', '退出', 'quit']:
                print("程序已退出。")
                break
            if user_input.strip():
                print("生成中...")
                try:
                    answer = generate_answer(user_input)
                    print("\n" + "="*40)
                    print("翻译结果:")
                    print("="*40)
                    print(answer)
                    print("="*40)
                except Exception as e:
                    print(f"生成过程中出错: {e}")
        except KeyboardInterrupt:
            print("\n程序已退出。")
            break
        except Exception as e:
            print(f"输入处理出错: {e}")

if __name__ == "__main__":
    main()