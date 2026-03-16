# NLP-Projects
基于 PyTorch 实现的 NLP 方向项目集合，聚焦**机器翻译、情感分析与大模型基础**，适配 RTX 系列显卡训练。

---

## 核心项目
### 1. Transformer 中英机器翻译模型
- **项目描述**：基于 Transformer Encoder-Decoder 架构重构中英翻译模型，解决传统 Seq2Seq 长文本依赖捕捉不足的问题；融入 GPT 风格 Top-K 解码策略，提升翻译流畅度。
- **技术栈**：Python / PyTorch / Transformer（多头注意力/位置编码）/ 文本预处理
- **量化成果**：
  - 模型 BLEU 值达 **42.5**，较原 Seq2Seq 模型提升 **36.2%**
  - 基于 RTX 3060 训练，batch_size=32，训练效率提升 **25%**
  - 长句（≥50词）翻译流畅度提升 **15%**
- **核心文件**：`model.py`（Transformer 模型实现）、`config.py`（参数配置）

---

### 2. 基于 LSTM+BERT 思路的文本情感分析系统
- **项目描述**：针对电商评论数据集，将 BERT「预训练+微调」思路融入 LSTM 模型，优化语义特征捕捉能力，提升情感分类准确率。
- **技术栈**：Python / PyTorch / LSTM / BERT / 文本预处理
- **量化成果**：
  - 模型准确率达 **90.7%**，较纯 LSTM 模型提升 **8.4%**
  - 歧义文本识别准确率从 **36%** 提升至 **89%**
- **核心文件**：`analysis-model.py`

---

## 快速开始
1. 克隆仓库：
   ```bash
   git clone https://github.com/paidaxing190/NLP-Projects.git
   cd NLP-Projects
