import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """监督对比损失，增强特征空间区分度"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]

        # 构建标签mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算相似度矩阵
        similarity = torch.matmul(features, features.T) / self.temperature

        # 排除自身对比
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob.mean()
