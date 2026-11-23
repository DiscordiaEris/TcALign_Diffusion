import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel


class CLIPEncoder(nn.Module):
    """CLIP编码器封装"""

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda", local_path=""):
        super().__init__()
        self.device = device
        self.model_name = model_name

        # 优先使用本地路径，否则从 HuggingFace Hub 加载
        load_from = local_path if local_path else model_name
        local_only = bool(local_path)

        # 加载 tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(load_from, local_files_only=local_only)

        # 加载完整 CLIP 模型以获取 visual_projection (旧版 HF 的 CLIPVisionModel 不含此层)
        full_clip = CLIPModel.from_pretrained(load_from, local_files_only=local_only)

        # 拆分为文本和视觉两部分
        self.text_encoder = full_clip.text_model.to(device)
        self.vision_encoder = full_clip.vision_model.to(device)
        self.visual_projection = full_clip.visual_projection.to(device)
        self.text_projection = full_clip.text_projection.to(device)

        # 冻结所有参数
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False
        for param in self.text_projection.parameters():
            param.requires_grad = False

        # 开启 gradient checkpointing: 反向传播时重算中间激活而非存储
        # 大幅降低 loss_clip 梯度链路的显存占用（24 层 ViT 激活量极大）
        if hasattr(self.vision_encoder, 'gradient_checkpointing_enable'):
            self.vision_encoder.gradient_checkpointing_enable()
            print("CLIP vision encoder: gradient checkpointing enabled")

        self.embed_dim = self.text_encoder.config.hidden_size  # 768 for ViT-L

        # CLIP 预处理: 先 Resize，再归一化 (CLIP 期望 [0, 1] 输入)
        import torchvision.transforms as T
        self.clip_preprocess = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # 清理不需要的全模型引用，释放内存
        del full_clip

    def encode_text(self, texts):
        """编码文本为CLIP嵌入"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        # Match CLIPModel.get_text_features: EOS pooled output followed by the
        # learned projection. Mean-pooling hidden states is not in CLIP's joint
        # image/text space.
        text_embeddings = self.text_projection(outputs.pooler_output)
        return torch.nn.functional.normalize(text_embeddings, dim=-1)

    def encode_image(self, images):
        """编码图像为CLIP嵌入 (输入范围: [-1, 1])

        使用 CLIP 预训练的 visual_projection，输出 768 维 CLIP 图像嵌入。
        流程: vision_encoder → pooler_output [1024] → visual_projection [768]
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # 从 [-1, 1] 反归一化到 [0, 1]，再应用 CLIP 预处理
        images_01 = (images + 1.0) / 2.0
        processed_images = self.clip_preprocess(images_01)

        outputs = self.vision_encoder(pixel_values=processed_images.to(self.device))
        # pooler_output: [batch, hidden_size], then project to joint space
        pooled = outputs.pooler_output
        image_embeddings = self.visual_projection(pooled)  # [batch, 768]

        return torch.nn.functional.normalize(image_embeddings, dim=-1)

    def forward(self, texts=None, images=None):
        """前向传播"""
        if texts is not None and images is not None:
            text_embeddings = self.encode_text(texts)
            image_embeddings = self.encode_image(images)
            return text_embeddings, image_embeddings
        elif texts is not None:
            return self.encode_text(texts)
        elif images is not None:
            return self.encode_image(images)
        else:
            raise ValueError("Either texts or images must be provided")
