#!/usr/bin/env python3
"""
Deep3DFR 集成验证脚本
检查所有必要的依赖、路径和文件是否就位
"""

import os
import sys

def verify_setup():
    """验证所有依赖是否就位"""
    errors = []
    warnings = []
    
    # 检查 Deep3DFR
    from configs.base_config import BaseConfig
    config = BaseConfig()
    
    print("="*70)
    print("🔍 Deep3DFR 集成验证")
    print("="*70)
    
    print("\n📂 检查 Deep3DFR 文件结构...")
    
    # 1. 检查 Deep3DFR 仓库
    if not os.path.isdir(config.deep3d_repo_path):
        errors.append(f"❌ Deep3DFR 目录不存在: {config.deep3d_repo_path}")
        print(f"   ❌ {config.deep3d_repo_path}")
    else:
        print(f"   ✅ Deep3DFR 代码: {config.deep3d_repo_path}")
        # 检查关键子目录
        required_dirs = ['models', 'util', 'BFM', 'checkpoints']
        for subdir in required_dirs:
            subpath = os.path.join(config.deep3d_repo_path, subdir)
            if os.path.isdir(subpath):
                print(f"      ✅ {subdir}/")
            else:
                warnings.append(f"⚠️  缺失子目录: {subpath}")
                print(f"      ⚠️  {subdir}/ (缺失)")
    
    # 2. 检查权重文件
    if not os.path.isfile(config.deep3d_checkpoint_path):
        errors.append(f"❌ 权重文件不存在: {config.deep3d_checkpoint_path}")
        print(f"   ❌ 权重文件: {config.deep3d_checkpoint_path}")
    else:
        file_size_mb = os.path.getsize(config.deep3d_checkpoint_path) / (1024**2)
        print(f"   ✅ 权重文件: {config.deep3d_checkpoint_path} ({file_size_mb:.1f} MB)")
    
    # 3. 检查 BFM 模型
    if not os.path.isfile(config.bfm_model_path):
        errors.append(f"❌ BFM 模型不存在: {config.bfm_model_path}")
        print(f"   ❌ BFM 模型: {config.bfm_model_path}")
    else:
        file_size_mb = os.path.getsize(config.bfm_model_path) / (1024**2)
        print(f"   ✅ BFM 模型: {config.bfm_model_path} ({file_size_mb:.1f} MB)")
    
    # 4. 检查表情基（可选）
    exp_basis_path = os.path.join(config.deep3d_repo_path, "BFM", "Exp_Pca.bin")
    if not os.path.isfile(exp_basis_path):
        warnings.append(f"⚠️  缺失表情基: {exp_basis_path}")
        print(f"   ⚠️  表情基: {exp_basis_path} (缺失，可能不必需)")
    else:
        print(f"   ✅ 表情基: {exp_basis_path}")
    
    # 5. 检查必要的 Python 包
    print("\n📦 检查 Python 依赖...")
    required_packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('pytorch3d', 'pytorch3d'),
        ('scipy', 'scipy'),
    ]
    
    missing_packages = []
    for pkg_display, import_name in required_packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {pkg_display:<20} (v{version})")
        except ImportError:
            errors.append(f"❌ 缺失包: {pkg_display}")
            print(f"   ❌ {pkg_display:<20}")
            missing_packages.append(pkg_display)
    
    # 6. 检查 3D 参数维度
    print("\n🎯 3D 参数配置:")
    print(f"   shape_dim: {config.shape_dim}")
    print(f"   expression_dim: {config.expression_dim}")
    print(f"   pose_dim: {config.pose_dim}")
    print(f"   texture_dim: {config.texture_dim}")
    print(f"   lighting_dim: {config.lighting_dim}")
    print(f"   total_3d_dim: {config.total_3d_dim}")
    
    # 7. 检查其他必要路径
    print("\n📁 其他配置:")
    print(f"   output_dir: {config.output_dir}")
    print(f"   checkpoint_dir: {config.checkpoint_dir}")
    print(f"   device: {config.device}")
    
    # 输出结果
    print("\n" + "="*70)
    if errors:
        print("❌ 发现以下关键错误:")
        for error in errors:
            print(f"   {error}")
        print("\n必须修复上述错误才能继续。")
        
        # 给出修复建议
        if missing_packages:
            print(f"\n💡 安装缺失的包:")
            print(f"   pip install {' '.join(missing_packages)}")
        
        if "权重文件不存在" in str(errors):
            print(f"\n💡 权重文件下载:")
            print(f"   Google Drive: https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP")
            print(f"   下载后放置到: {config.deep3d_checkpoint_path}")
        
        if "BFM 模型不存在" in str(errors):
            print(f"\n💡 BFM 模型下载:")
            print(f"   申请链接: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads")
            print(f"   文件: 01_MorphableModel.mat")
            print(f"   放置到: {config.bfm_model_path}")
        
        return False
    else:
        print("✅ 所有检查通过！")
        if warnings:
            print("\n⚠️  警告:")
            for warning in warnings:
                print(f"   {warning}")
        print("\n可以开始训练或推理了！")
        return True

if __name__ == "__main__":
    try:
        success = verify_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 验证过程出错:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
