import torch
import torch.nn as nn


class AttributeEmbeddingSpace(nn.Module):
    """属性嵌入空间 (AES) 模块"""

    def __init__(self, embed_dim=512, attribute_dims=None):
        """
        attribute_dims: 一个字典，定义每个属性子空间的基向量数量。
                        例如: {'expression': 5, 'pose': 4, 'lighting': 4}
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.attribute_dims = attribute_dims if attribute_dims else {
            'expression': 5,
            'pose': 4,
            'lighting': 4
        }

        # 为每个属性创建一个可学习的基向量矩阵
        self.attribute_bases = nn.ParameterDict()
        for attr, dim in self.attribute_dims.items():
            # 每个属性有dim个基向量，每个基向量是embed_dim维
            basis = nn.Parameter(torch.randn(dim, embed_dim))
            nn.init.normal_(basis, std=0.02)
            self.attribute_bases[attr] = basis

    def project_to_subspace(self, embedding, attribute):
        """
        将嵌入向量投影到某个特定属性子空间
        返回: 投影后的向量, 残差向量
        """
        with torch.cuda.amp.autocast(enabled=False):
            basis = self.attribute_bases[attribute].float()
            orthogonal, _ = torch.linalg.qr(basis.T, mode="reduced")
            projected = (embedding.float() @ orthogonal) @ orthogonal.T
        return projected, embedding - projected

    def manipulate(self, e_img, e_txt, target_attributes):
        """
        根据目标文本嵌入，在指定的属性子空间中操控图像嵌入
        target_attributes: 要操作的属性列表，例如 ['expression', 'pose'] 或字符串
        支持批处理：处理不同样本的不同属性
        """
        batch_size = len(e_img)
        if isinstance(target_attributes, str):
            target_attributes = [[target_attributes]] * batch_size
        elif len(target_attributes) == 1 and isinstance(target_attributes[0], str):
            target_attributes = [target_attributes] * batch_size
        elif target_attributes and isinstance(target_attributes[0], str):
            target_attributes = [[attribute] for attribute in target_attributes]
        if len(target_attributes) != batch_size:
            raise ValueError("Expected one attribute selection per input image")

        outputs = []
        with torch.cuda.amp.autocast(enabled=False):
            difference = e_txt.float() - e_img.float()
            bases = {}
            for index, selected in enumerate(target_attributes):
                key = tuple(dict.fromkeys(attr for attr in selected if attr in self.attribute_bases))
                if not key:
                    outputs.append(e_img[index].float())
                    continue
                if key not in bases:
                    basis = torch.cat([self.attribute_bases[attr] for attr in key]).float()
                    bases[key], _ = torch.linalg.qr(basis.T, mode="reduced")
                basis = bases[key]
                outputs.append(e_img[index].float() + (difference[index] @ basis) @ basis.T)
        return torch.stack(outputs)
