from typing import Any
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn


class ProtoSoftmaxLayer(nn.Module):
    def __init__(self, config: Any, sentence_encoder: nn.Module, num_classes: int):
        """
        初始化ProtoSoftmaxLayer。

        参数:
        config (object) -- 配置对象，包含模型的超参数和配置信息
        sentence_encoder (nn.Module) -- 句子编码器（通常是BERT或其他文本编码模型）
        num_classes (int) -- 类别数，定义输出层的大小
        """
        super(ProtoSoftmaxLayer, self).__init__()

        self.prototypes = None  # 初始化原型（在增量学习中会更新）
        self.config = config  # 配置对象
        self.sentence_encoder = sentence_encoder  # 句子编码器
        self.num_classes = num_classes  # 类别数
        self.hidden_size = self.sentence_encoder.get_output_size()  # 获取句子编码器的输出维度（隐藏层大小）
        self.classifier = nn.Linear(self.hidden_size, self.num_classes, bias=False)  # 全连接层，无偏置

    @staticmethod
    def __calculate_distance__(representation: torch.Tensor, prototypes: torch.Tensor,
                               alpha: float = 0.5) -> torch.Tensor:
        """
        混合点积相似度与余弦相似度

        参数：
        alpha : 混合比例 (0.0=纯点积, 1.0=纯余弦)
        """
        dot_product = torch.matmul(representation, prototypes.T)  # (B, C)

        norm_rep = torch.norm(representation, p=2, dim=-1, keepdim=True)  # (B, 1)
        norm_proto = torch.norm(prototypes, p=2, dim=-1)  # (C,)

        cosine_sim = dot_product / (norm_rep * norm_proto + 1e-8)  # (B, C)

        return (1 - alpha) * dot_product + alpha * cosine_sim

    def memory_forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        通过内存原型计算距离。

        参数:
        representation (torch.Tensor) -- 输入表示，形状为(batch_size, hidden_size)

        返回:
        torch.Tensor -- 距离矩阵，形状为(batch_size, num_classes)
        """
        distance_memory = self.__calculate_distance__(representation, self.prototypes)  # 计算输入与原型的距离
        return distance_memory

    def set_memorized_prototypes(self, prototypes: torch.Tensor) -> None:
        """
        设置和存储原型。

        参数:
        prototypes (torch.Tensor) -- 存储的原型，形状为(num_classes, hidden_size)
        """
        # 存储原型并将其转移到指定设备上
        self.prototypes = prototypes.detach().to(self.config.device)

    def get_feature(self, sentences: torch.Tensor) -> torch.Tensor:
        """
        获取句子的表示特征。

        参数:
        sentences (torch.Tensor) -- 输入的句子，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 获取到的句子特征，形状为(batch_size, hidden_size)
        """
        representation = self.sentence_encoder(sentences)  # 使用句子编码器获取句子特征
        return representation.cpu().data.numpy()  # 返回CPU上的特征，转换为numpy数组

    def get_mem_feature(self, representation: torch.Tensor) -> torch.Tensor:
        """
        获取表示与原型的距离特征。

        参数:
        representation (torch.Tensor) -- 输入的表示，形状为(batch_size, hidden_size)

        返回:
        torch.Tensor -- 计算得到的距离特征，形状为(batch_size, num_classes)
        """
        distance = self.memory_forward(representation)  # 计算输入表示与原型的距离
        return distance

    def incremental_learning(self, old_class_count: int, new_class_count: int) -> None:
        """
        执行增量学习，增加类别数并调整全连接层的权重。

        参数:
        old_class_count (int) -- 旧的类别数
        new_class_count (int) -- 新增的类别数
        """
        # 获取当前全连接层的权重
        weight = self.classifier.weight.data
        # 更新全连接层，新的类别数为原类别数 + 新增类别数
        self.classifier = nn.Linear(768, old_class_count + new_class_count, bias=False).to(self.config.device)

        with torch.no_grad():
            # 将旧类别的权重保留到新的全连接层中
            self.classifier.weight.data[:old_class_count] = weight[:old_class_count]

    @staticmethod
    def _init_weights(module: nn.Module):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def forward(self, sentences: torch.Tensor) -> tuple[Any, Any]:
        """
        前向传播，通过句子编码器获取句子表示，并通过全连接层计算logits。

        参数:
        sentences (torch.Tensor) -- 输入的句子，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- logits和句子表示，形状为(batch_size, num_classes)和(batch_size, hidden_size)
        """
        # 获取句子表示
        representation = self.sentence_encoder(sentences)  # (B, H)
        logits = self.classifier(representation)  # 通过全连接层计算logits
        return logits, representation  # 返回logits和表示


class BertEncoder(nn.Module):
    def __init__(self, config: Any):
        """
        初始化BertEncoder模型。

        参数:
        config (object) -- 配置对象，包含模型的超参数和配置信息
        """
        super(BertEncoder, self).__init__()

        # 从预训练模型路径加载配置，并设置mean_resizing为False
        self.bert_config = AutoConfig.from_pretrained(config.bert_model_path, mean_resizing=False)

        # 加载预训练BERT模型
        self.encoder = AutoModel.from_pretrained(config.bert_model_path, config=self.bert_config)

        # 设置编码模式（'standard'或'entity_marker'）
        self._setup_encoding_pattern(config)

    def _setup_encoding_pattern(self, config: Any) -> None:
        """
        根据配置设置编码模式。

        参数:
        config (object) -- 配置对象，包含编码模式及其他设置

        返回:
        None
        """
        # 验证传入的编码模式是否合法
        if config.encoding_mode not in ['standard', 'entity_marker']:
            raise ValueError(f"Invalid pattern: {config.encoding_mode}")

        # 设置编码模式和输出维度
        self.pattern = config.encoding_mode
        self.output_size = 768  # 输出维度，通常为BERT的hidden_size

        # 如果编码模式为'entity_marker'，调整BERT词汇表大小并设置线性变换
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)  # 增加4个标记（实体标记）
            self.linear_transform = nn.Sequential(
                nn.Linear(self.bert_config.hidden_size * 2, self.bert_config.hidden_size, bias=True),  # 合并两个实体向量后的线性变换
                nn.GELU(),  # GELU激活函数
                nn.LayerNorm([self.bert_config.hidden_size])  # 层归一化
            )
        else:
            # 如果编码模式为'标准'，只进行一个简单的线性变换
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size)

        # 初始化权重
        self._init_weights()

        # Dropout层用于防止过拟合
        self.drop = nn.Dropout(config.dropout_rate)

    def _init_weights(self) -> None:
        """
        初始化线性层的权重和偏置。

        返回:
        None
        """
        for module in self.linear_transform.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)  # 使用Xavier初始化
                nn.init.constant_(module.bias, 0)  # 偏置初始化为0

    def get_output_size(self) -> int:
        """
        获取输出的维度大小。

        返回:
        int -- 输出维度大小
        """
        return self.output_size

    def _standard_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用标准的BERT输出方式进行前向传播。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- BERT池化后的输出，形状为(batch_size, hidden_size)
        """
        outputs = self.encoder(input_ids)  # BERT模型的前向输出
        return outputs.pooler_output  # 返回池化后的输出（CLS标记的表示）

    def _entity_marker_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用实体标记编码方式进行前向传播。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 融合后的实体向量，形状为(batch_size, hidden_size)
        """
        # 创建注意力掩码（非零位置为有效）
        attention_mask = input_ids != 0

        # 进行BERT前向传播
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # 获取BERT最后一层的隐藏状态

        # 获取实体标记的位置，30522和30524是特殊的实体标记ID
        e11_pos = (input_ids == 30522).int().argmax(dim=1)  # 获取第一个实体标记的位置
        e21_pos = (input_ids == 30524).int().argmax(dim=1)  # 获取第二个实体标记的位置

        # 创建批次索引，用于从序列输出中获取对应位置的向量
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        # 获取对应位置的实体向量
        e11_vectors = sequence_output[batch_indices, e11_pos]
        e21_vectors = sequence_output[batch_indices, e21_pos]

        # 将两个实体向量连接（拼接）
        combined = torch.cat([e11_vectors, e21_vectors], dim=1)

        # 通过线性变换和Dropout层进行处理
        hidden = self.linear_transform(self.drop(combined))

        return hidden

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        根据编码模式选择对应的前向传播方式。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 模型的最终输出，形状为(batch_size, output_size)
        """
        if self.pattern == 'standard':
            return self._standard_forward(input_ids)
        return self._entity_marker_forward(input_ids)
