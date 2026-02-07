import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd
import os
import json
from typing import List, Dict, Optional

class ESAModule(nn.Module):
    """
    Embedded Semantic Association Module (ESAM)
    使用注意力机制关联句子嵌入和描述嵌入
    """
    def __init__(self, hidden_size: int, num_classes: int = 50):
        super(ESAModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        # 全连接层映射到分类输出
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, sentence_emb: torch.Tensor, description_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence_emb: [batch_size, hidden_size] - 句子嵌入
            description_emb: [batch_size, hidden_size] - 描述嵌入

        Returns:
            logits: [batch_size, num_classes] - 分类logits
        """
        # 确保输入是3D: [batch_size, seq_len, hidden_size]
        # 句子和描述都是单个向量，所以seq_len=1
        sentence_emb = sentence_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]
        description_emb = description_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 第一步注意力: Es作为Q和V, Ed作为K
        z1, _ = self.attention(sentence_emb, description_emb, sentence_emb)  # [batch_size, 1, hidden_size]

        # 第二步注意力: Ed作为Q和V, Es作为K
        z2, _ = self.attention(description_emb, sentence_emb, description_emb)  # [batch_size, 1, hidden_size]

        # 拼接z1和z2
        combined = torch.cat([z1.squeeze(1), z2.squeeze(1)], dim=-1)  # [batch_size, hidden_size * 2]

        # 通过分类器
        logits = self.classifier(combined)  # [batch_size, num_classes]

        return logits

class ESAMBertForSequenceClassification(nn.Module):
    """
    基于BERT的ESAM模型，用于序列分类
    结合句子嵌入和描述嵌入，通过ESAM模块进行关联
    """
    def __init__(self, model_name_or_path: str, num_classes: int = 50, description_embeddings: Optional[Dict[str, torch.Tensor]] = None):
        super(ESAMBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes

        # BERT模型
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size

        # ESAM模块
        self.esam = ESAModule(hidden_size=self.hidden_size, num_classes=num_classes)

        # 分类头（可选，用于传统分类）
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # 描述嵌入字典
        self.description_embeddings = description_embeddings or {}

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, description_emb=None, use_esam=True):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] - 用于计算损失
            description_emb: [batch_size, hidden_size] - 预计算的描述嵌入
            use_esam: 是否使用ESAM模块

        Returns:
            outputs: 包含loss, logits等
        """
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sentence_emb = outputs.pooler_output  # [batch_size, hidden_size] - 使用pooler_output作为句子嵌入

        if use_esam and description_emb is not None:
            # 使用ESAM
            logits = self.esam(sentence_emb, description_emb)
        else:
            # 传统分类
            pooled_output = self.dropout(sentence_emb)
            logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 如果labels是one-hot编码，转换为类别索引
            if labels.dim() > 1 and labels.shape[-1] == self.num_classes:
                labels = labels.argmax(dim=-1)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'sentence_emb': sentence_emb,
            'description_emb': description_emb
        }

    def get_description_embedding(self, label: str, tokenizer: BertTokenizer) -> torch.Tensor:
        """
        获取标签对应的描述嵌入
        如果有多个描述，取平均
        """
        if label in self.description_embeddings:
            return self.description_embeddings[label]

        # 如果没有预计算，返回零向量（实际使用时应该预计算）
        return torch.zeros(self.hidden_size)

class ESAMDataProcessor:
    """
    ESAM数据处理器
    处理描述数据的预计算和加载
    """
    def __init__(self, description_file: str, tokenizer: BertTokenizer, model: BertModel):
        self.description_file = description_file
        self.tokenizer = tokenizer
        self.model = model
        self.device = next(model.parameters()).device
        self.description_embeddings = {}

        # 预计算描述嵌入
        self._precompute_description_embeddings()

    def _precompute_description_embeddings(self):
        """预计算所有标签的描述嵌入"""
        print("预计算描述嵌入...")

        # 读取描述文件
        if self.description_file.endswith('.csv'):
            df = pd.read_csv(self.description_file)
            descriptions = {}
            for _, row in df.iterrows():
                label = row['ID']
                desc = row['description']
                if label not in descriptions:
                    descriptions[label] = []
                descriptions[label].append(desc)
        else:
            # 假设是JSON或其他格式
            with open(self.description_file, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)

        # 对每个标签计算嵌入
        for label, desc_list in descriptions.items():
            if isinstance(desc_list, str):
                desc_list = [desc_list]

            # Tokenize所有描述
            tokenized = self.tokenizer(desc_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**tokenized)
                embeddings = outputs.pooler_output  # [num_descriptions, hidden_size]

            # 平均多个描述的嵌入
            avg_embedding = embeddings.mean(dim=0)  # [hidden_size]

            self.description_embeddings[label] = avg_embedding.cpu()

        print(f"预计算完成，共 {len(self.description_embeddings)} 个标签的描述嵌入")

    def get_description_embedding(self, label: str) -> torch.Tensor:
        """获取标签的描述嵌入"""
        return self.description_embeddings.get(label, torch.zeros(self.model.config.hidden_size))

# 使用示例
if __name__ == "__main__":
    # 初始化tokenizer和基础模型
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    base_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')

    # 初始化数据处理器
    processor = ESAMDataProcessor(
        description_file=r'smallData/technique_description_f.csv',
        tokenizer=tokenizer,
        model=base_model
    )

    # 创建ESAM模型
    model = ESAMBertForSequenceClassification(
        model_name_or_path='allenai/scibert_scivocab_uncased',
        num_classes=50,
        description_embeddings=processor.description_embeddings
    )

    print("ESAM模型创建成功")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    