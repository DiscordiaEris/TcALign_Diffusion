import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_image_grid(original_images, generated_images, texts, output_path):
    """保存图像网格"""
    # 将图像从[-1, 1]转换到[0, 1]
    original_images = (original_images + 1) / 2
    generated_images = (generated_images + 1) / 2

    # 创建图像网格
    grid = make_grid(torch.cat([original_images, generated_images], dim=0), nrow=len(original_images))

    # 保存图像
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

    # 添加文本描述
    for i, text in enumerate(texts):
        plt.figtext(0.5, 0.95 - i * 0.05, text, ha="center", fontsize=10,
                    bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()