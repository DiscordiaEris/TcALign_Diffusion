import torch
import torch.nn as nn
import torch.nn.functional as F


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
        basis = self.attribute_bases[attribute]  # [dim, embed_dim]

        # 计算投影系数: embedding * basis^T
        coeff = torch.matmul(embedding, basis.T)  # [batch_size, dim]

        # 重建投影后的向量: coeff * basis
        proj_embedding = torch.matmul(coeff, basis)  # [batch_size, embed_dim]

        # 计算残差
        residual = embedding - proj_embedding

        return proj_embedding, residual

    def manipulate(self, e_img, e_txt, target_attributes):
        """
        根据目标文本嵌入，在指定的属性子空间中操控图像嵌入
        target_attributes: 要操作的属性列表，例如 ['expression', 'pose'] 或字符串
        支持批处理：处理不同样本的不同属性
        """
        batch_size = e_img.shape[0]
        
        # 若传入字符串，转为列表
        if isinstance(target_attributes, str):
            target_attributes = [target_attributes]
        
        # 若列表长度为1，广播到batch_size
        if len(target_attributes) == 1 and isinstance(target_attributes[0], str):
            target_attributes = target_attributes * batch_size
        
        # 若是列表的列表（每个样本一个属性列表），逐样本处理
        if isinstance(target_attributes[0], list):
            manipulated_embeddings = []
            for i, attr_list in enumerate(target_attributes):
                manipulated_emb = e_img[i:i+1].clone()
                for attr in attr_list:
                    if attr not in self.attribute_dims:
                        continue
                    proj_img, res_img = self.project_to_subspace(manipulated_emb, attr)
                    proj_txt, _ = self.project_to_subspace(e_txt[i:i+1], attr)
                    manipulated_emb = proj_txt + res_img
                manipulated_embeddings.append(manipulated_emb)
            return torch.cat(manipulated_embeddings, dim=0)
        
        # 若是字符串列表，逐样本应用相同属性
        manipulated_embeddings = []
        for i, attr in enumerate(target_attributes):
            manipulated_emb = e_img[i:i+1].clone()
            if isinstance(attr, str) and attr in self.attribute_dims:
                proj_img, res_img = self.project_to_subspace(manipulated_emb, attr)
                proj_txt, _ = self.project_to_subspace(e_txt[i:i+1], attr)
                manipulated_emb = proj_txt + res_img
            manipulated_embeddings.append(manipulated_emb)
        
        return torch.cat(manipulated_embeddings, dim=0)