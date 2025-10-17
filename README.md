# 法律术语翻译模型 - LoRA微调实践


## 项目概述


本项目基于Qwen1.5-0.5B模型，使用LoRA技术对"法律术语翻译成大白话"这一特定任务进行参数高效微调。通过构建高质量的50条训练数据和5条验证数据，实现了在法律术语解释任务上的显著效果提升。


## 设计哲学


### 架构设计

```
项目架构：
├── 数据层 (Data Layer)
│   ├── finetune_dataset.json (50条训练数据)
│   └── 数据预处理模块
├── 模型层 (Model Layer)  
│   ├── Qwen1.5-0.5B基础模型
│   └── LoRA适配器
├── 训练层 (Training Layer)
│   ├── LoRA微调训练脚本
│   └── 训练监控与评估
└── 推理层 (Inference Layer)
    ├── 模型加载与推理
    └── 交互式测试界面
```


### 关键技术选型及原因


1. **基础模型选择：Qwen1.5-0.5B**

   - **原因**：参数量适中(0.5B)，在消费级GPU上可快速训练，中文理解能力强，开源可商用
   - **对比考虑**：相比更大的模型(如1.8B/7B)，训练成本更低，推理速度更快


2. **微调方法：LoRA (Low-Rank Adaptation)**

   - **原因**：参数高效，只需训练少量参数(约0.67%)，大幅降低显存需求
   - **优势**：训练速度快，可移植性强，多个任务可共享基础模型


3. **训练框架：Transformers + PEFT**

   - **原因**：Hugging Face生态成熟，社区支持好，代码可维护性强
   - **优势**：标准化接口，易于扩展和部署


4. **数据格式：Instruction-Input-Output**

   - **原因**：符合指令微调范式，明确任务指令，提升模型理解能力
   - **优势**：格式统一，便于数据管理和质量控制


## 环境与运行


### 系统要求

- GPU: NVIDIA RTX 3090/4090/5090 (至少8GB显存)
- 内存: 16GB RAM
- 存储: 至少10GB可用空间


### 环境配置


1. **安装依赖**：

```bash
pip install torch torchvision torchaudio --index-url `https://download.pytorch.org/whl/cu118`  
pip install transformers==4.37.0 peft==0.8.0 datasets==2.14.0 accelerate==0.25.0
pip install numpy<2 pandas==2.2.2 pyarrow==14.0.2
```


2. **验证安装**：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers, peft, datasets; print('所有依赖安装成功!')"
```


### 项目运行指南


#### 步骤1：准备模型文件

确保 `qwen1.5-0.5b-model/` 目录包含基础模型文件。


#### 步骤2：数据准备

项目已包含50条训练数据：

```bash
ls -la finetune_dataset.json
```


#### 步骤3：模型训练

```bash
# 运行优化版本的训练（推荐）
python train_optimized.py
```


训练过程监控：

- GPU使用情况
- 训练损失曲线
- 验证集表现
- 模型保存进度


#### 步骤4：模型推理

```bash
# 交互式测试
python run_optimized.py
```


#### 步骤5：效果评估

```bash
# 运行评估脚本对比基础模型和微调模型
python evaluate_model.py
```


## 成果展示


### 核心成果复现


#### 1. 快速测试

```bash
# 直接使用训练好的模型进行测试
python run_optimized.py
```


输入测试样例：

```
> 原告应承担举证责任，否则将面临败诉风险。
```


预期输出：

```
谁告状谁就得拿出证据来证明自己说得对，要是拿不出证据，很可能就会输掉官司。
```


#### 2. 批量测试

```bash
# 运行预设的测试用例
python test_model.py
```


#### 3. 完整训练复现

```bash
# 清理之前的训练结果
rm -rf legal_translate_optimized/

# 重新开始训练
python train_optimized.py
```


### 性能指标


- **训练时间**: ~10分钟 (RTX 5090)
- **模型大小**: 基础模型1.1GB + LoRA权重15MB
- **可训练参数**: 1.6% (786,432 / 464,774,144)
- **推理速度**: ~2秒/条


### 效果对比


**基础模型（Zero-shot）**：

```
输入: "原告应承担举证责任，否则将面临败诉风险。"
输出: "在法律诉讼中，提出诉讼的一方需要提供证据来支持其主张，如果无法提供足够的证据，可能会面临败诉的风险。"
```


**微调模型（LoRA）**：

```
输入: "原告应承担举证责任，否则将面临败诉风险。"
输出: "谁告状谁就得拿出证据来证明自己说得对，要是拿不出证据，很可能就会输掉官司。"
```


## 项目结构

```
legal_translation_lora/
├── README.md
├── requirements.txt
├── finetune_dataset.json          # 50条训练数据
├── train_autodl.py               # 基础训练脚本
├── train_optimized.py            # 优化训练脚本
├── run_autodl.py                 # 基础推理脚本
├── run_optimized.py              # 优化推理脚本
├── local_dataset.py              # 数据加载模块
├── test_model.py                 # 测试脚本
├── evaluate_model.py             # 评估脚本
├── download_model.py             # 模型下载脚本
└── legal_translate_optimized/    # 训练好的模型
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```


## 文件说明


### 核心脚本

- `train_optimized.py` - 优化训练脚本，包含最佳参数配置
- `run_optimized.py` - 优化推理脚本，支持交互式测试
- `local_dataset.py` - 数据加载和处理模块
- `evaluate_model.py` - 模型效果对比评估


### 数据文件

- `finetune_dataset.json` - 50条高质量法律术语翻译数据


### 模型文件

- `qwen1.5-0.5b-model/` - 基础模型（需自行准备）
- `legal_translate_optimized/` - 微调后的LoRA权重


## 技术亮点


1. **参数高效**: 使用LoRA技术，仅训练0.67%的参数
2. **快速训练**: 在RTX 5090上10分钟完成训练
3. **效果显著**: 微调后模型输出更符合"大白话"要求
4. **易于部署**: 小体积LoRA权重便于分发和更新
5. **可扩展性**: 模块化设计支持其他法律细分领域微调


## 评估方法


### 定量评估

运行评估脚本生成对比结果：

```bash
python evaluate_model.py
```


### 定性评估

从三个维度人工评估：

1. **忠实原文** - 解释是否准确反映法律含义 (1-5分)
2. **通俗易懂** - 是否使用普通人能理解的语言 (1-5分)  
3. **表达自然** - 语言是否流畅自然 (1-5分)


## 常见问题


### Q: 训练过程中出现GPU内存不足

A: 减小 `per_device_train_batch_size` 或 `max_seq_length`


### Q: 模型下载失败

A: 使用国内镜像或手动下载模型文件


### Q: 训练效果不理想

A: 增加训练轮次 `num_train_epochs` 或调整学习率 `learning_rate`


## 后续优化方向


1. 增加训练数据量和多样性
2. 尝试QLoRA进一步降低显存需求  
3. 引入强化学习优化输出质量
4. 支持批量处理和API服务化


## 许可证


本项目采用MIT许可证。模型权重遵循原始模型的许可证要求。


## 贡献


欢迎提交Issue和Pull Request来改进本项目。