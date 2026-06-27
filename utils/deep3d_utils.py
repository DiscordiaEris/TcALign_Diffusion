import torch
import numpy as np
import sys
import os
import cv2
import importlib
from PIL import Image


def _import_deep3d_modules(repo_path):
    """按配置路径动态导入 Deep3DFR 模块。

    本项目有 models/ 包，与 Deep3DFR/models/ 冲突，且 Deep3DFR 依赖
    arcface_torch 和 kornia（仅训练用，推理不需要）。
    这里做临时 sys.modules 置换 + arcface_torch mock 来绕过。
    """
    abs_repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(abs_repo_path):
        raise FileNotFoundError(
            f"Deep3DFR repo not found: {abs_repo_path}. "
            "Please set config.deep3d_repo_path to a valid directory."
        )

    # numpy 兼容性补丁：新版 numpy 删除了 VisibleDeprecationWarning
    if not hasattr(np, 'VisibleDeprecationWarning'):
        np.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (Warning,), {})

    # 放到 sys.path 最前
    if abs_repo_path in sys.path:
        sys.path.remove(abs_repo_path)
    sys.path.insert(0, abs_repo_path)

    # arcface_torch 和 kornia 的 mock：我们只用推理不需要人脸识别/几何变换
    for _mod_name in ('models.arcface_torch', 'models.arcface_torch.backbones'):
        if _mod_name not in sys.modules:
            sys.modules[_mod_name] = type(sys)(_mod_name)
    sys.modules['models.arcface_torch.backbones'].get_model = lambda **kw: None

    # kornia mock（仅当未安装时）
    if 'kornia' not in sys.modules:
        _kornia = type(sys)('kornia')
        _kornia_geom = type(sys)('kornia.geometry')
        _kornia_geom.warp_affine = lambda *a, **kw: None
        _kornia_cam = type(sys)('kornia.geometry.camera')
        _kornia_cam.pixel2cam = lambda *a, **kw: None
        sys.modules['kornia'] = _kornia
        sys.modules['kornia.geometry'] = _kornia_geom
        sys.modules['kornia.geometry.camera'] = _kornia_cam

    # 暂存并移出可能冲突的缓存模块
    _cached = {}
    for _key in ('models', 'util', 'options'):
        if _key in sys.modules:
            _cached[_key] = sys.modules.pop(_key)

    try:
        networks = importlib.import_module('models.networks')
        preprocess = importlib.import_module('util.preprocess')
        load_mats = importlib.import_module('util.load_mats')
    finally:
        # 恢复被移出的缓存模块（保留 mock 的 arcface_torch 避免后续错误）
        for _key, _mod in _cached.items():
            sys.modules[_key] = _mod

    # 把 Deep3DFR 从 sys.path 头移出，避免后续干扰
    if sys.path[0] == abs_repo_path:
        sys.path.pop(0)

    # skimage.io 无冲突
    io = importlib.import_module('skimage.io')
    return networks, preprocess.align_img, load_mats.load_lm3d, io


class Deep3DProcessor:
    """Deep3DFaceReconstruction 处理器"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 通过配置路径加载 Deep3DFR 依赖
        self.networks, self.align_img, self.load_lm3d, self.io = _import_deep3d_modules(
            config.deep3d_repo_path
        )

        # 加载模型
        self.model = self.load_model()

        # 加载 3DMM 模型和 landmarks
        self.lm3d_std = self.load_lm3d(os.path.join(config.deep3d_repo_path, 'BFM'))

    def load_model(self):
        """加载 Deep3DFaceReconstruction 模型 (FAN 网络)"""
        if not os.path.isfile(self.config.deep3d_checkpoint_path):
            raise FileNotFoundError(
                f"Deep3DFR checkpoint not found: {self.config.deep3d_checkpoint_path}. "
                "Please set config.deep3d_checkpoint_path to a valid .pth file."
            )

        # 创建 FAN 重建网络 (ResNet50 骨干)
        model = self.networks.define_net_recon(net_recon='resnet50', use_last_fc=False)

        # 加载训练好的权重 (处理 DataParallel 前缀和完整训练 checkpoint)
        raw_state_dict = torch.load(self.config.deep3d_checkpoint_path, map_location='cpu')

        # 判断是否为完整训练 checkpoint 格式: {'net_recon': {...}, 'opt_00': {...}, ...}
        if isinstance(raw_state_dict, dict) and 'net_recon' in raw_state_dict:
            print("Detected full training checkpoint, extracting model weights from 'net_recon' key")
            state_dict = raw_state_dict['net_recon']
        else:
            state_dict = raw_state_dict

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)

        model = model.to(self.device)
        model.eval()
        return model

    def process(self, image_tensor):
        """
        处理图像并返回3D参数

        参数:
            image_tensor: 输入图像张量 [1, 3, H, W], 值范围 [-1, 1]

        返回:
            params: 3D参数字典
        """
        # 将图像从 [-1, 1] 转换到 [0, 255] (BGR格式)
        image_np = ((image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        image_np = image_np[:, :, ::-1]  # RGB -> BGR

        # 保存临时图像文件
        temp_path = os.path.join(self.config.output_dir, "temp_input.jpg")
        cv2.imwrite(temp_path, image_np)

        # 使用 Deep3DFaceReconstruction 的处理流程
        params = self._reconstruct_face(temp_path)

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return params

    def _reconstruct_face(self, image_path):
        """重建3D人脸并返回参数"""
        # 读取图像 (skimage.io.imread 返回 numpy array, BGR 或 RGB)
        img_np = self.io.imread(image_path)
        # 转为 PIL.Image (align_img 需要 PIL.Image)
        from PIL import Image as PILImage
        if img_np.shape[-1] == 3:
            img = PILImage.fromarray(img_np)
        else:
            img = PILImage.fromarray(img_np)

        lm = self._detect_landmarks(img_np)

        if lm is None:
            raise ValueError("No face detected in the image")

        # 对齐图像 (返回: trans_params, img_new(PIL), lm_new, mask_new(PIL))
        trans_params, img_aligned_pil, lm_aligned, mask_new = self.align_img(
            img, lm, self.lm3d_std
        )

        # PIL.Image → tensor, 归一化到 [0, 1]
        img_aligned_np = np.array(img_aligned_pil, dtype=np.float32) / 255.0
        img_aligned = torch.from_numpy(
            img_aligned_np.transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)

        # 模型推理 (输出: [batch, 257] 展平向量)
        with torch.no_grad():
            coeffs = self.model(img_aligned)

        # 拆分 257 维参数向量 (Deep3DFR split_coeff 格式)
        # id:80, exp:64, tex:80, angle:3, gamma:27, trans:3 = 257
        coeffs_np = coeffs.squeeze(0).cpu().numpy()
        params = {
            'shape': coeffs_np[0:80].reshape(1, -1),
            'expression': coeffs_np[80:144].reshape(1, -1),
            'pose': coeffs_np[224:227].reshape(1, -1),
            'texture': coeffs_np[144:224].reshape(1, -1),
            'lighting': coeffs_np[227:254].reshape(1, -1),
            'camera': coeffs_np[254:257].reshape(1, -1),
        }

        return params

    def _detect_landmarks(self, img):
        """检测5个人脸关键点 (左眼, 右眼, 鼻子, 左嘴角, 右嘴角)

        尝试多种后端: dlib → mediapipe → MTCNN → OpenCV Haar
        """
        # 尝试 dlib (68点 → 5点转换)
        lm = self._detect_with_dlib(img)
        if lm is not None:
            return lm

        # 尝试 mediapipe (468点 → 5点转换)
        lm = self._detect_with_mediapipe(img)
        if lm is not None:
            return lm

        # 尝试 MTCNN (直接输出5点)
        lm = self._detect_with_mtcnn(img)
        if lm is not None:
            return lm

        # 最后尝试 OpenCV Haar (精度最低)
        lm = self._detect_with_opencv(img)
        if lm is not None:
            return lm

        return None

    def _detect_with_dlib(self, img):
        """使用 dlib 68点检测器"""
        try:
            import dlib
        except ImportError:
            return None

        try:
            detector = dlib.get_frontal_face_detector()
            predictor_path = os.path.join(
                self.config.deep3d_repo_path, "checkpoints",
                "shape_predictor_68_face_landmarks.dat"
            )
            if not os.path.isfile(predictor_path):
                return None

            predictor = dlib.shape_predictor(predictor_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)

            if len(faces) == 0:
                return None

            shape = predictor(gray, faces[0])
            pts_68 = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            # 68点 → 5点 (Deep3DFR 格式: left_eye, right_eye, nose, left_mouth, right_mouth)
            left_eye = pts_68[36:42].mean(axis=0)
            right_eye = pts_68[42:48].mean(axis=0)
            nose = pts_68[30]
            left_mouth = pts_68[48]
            right_mouth = pts_68[54]
            return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])

        except Exception:
            return None

    def _detect_with_mediapipe(self, img):
        """使用 mediapipe FaceMesh 检测关键点"""
        try:
            import mediapipe as mp
        except ImportError:
            return None

        try:
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            ) as face_mesh:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if not results.multi_face_landmarks:
                    return None

                h, w = img.shape[:2]
                lm = results.multi_face_landmarks[0]
                pts_all = np.array([[p.x * w, p.y * h] for p in lm.landmark])

                # MediaPipe 468点 → 5点
                # 索引参考: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
                left_eye = pts_all[[33, 133, 157, 158, 159, 160, 161, 246]].mean(axis=0)
                right_eye = pts_all[[362, 263, 387, 386, 385, 384, 398, 466]].mean(axis=0)
                nose = pts_all[1]
                left_mouth = pts_all[61]
                right_mouth = pts_all[291]
                return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])

        except Exception:
            return None

    def _detect_with_mtcnn(self, img):
        """使用 MTCNN 检测5点关键点"""
        try:
            from facenet_pytorch import MTCNN
        except ImportError:
            return None

        try:
            mtcnn = MTCNN(select_largest=True, device=self.device)
            # MTCNN 需要 PIL Image
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return None

            # points shape: (5, 2) — left_eye, right_eye, nose, left_mouth, right_mouth
            lm = points[0].T  # (2, 5) → (5, 2)
            return lm.astype(np.float32)

        except Exception:
            return None

    def _detect_with_opencv(self, img):
        """使用 OpenCV Haar 级联进行粗略估计 (最后备选方案)"""
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None

            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]

            # 在面部区域检测眼睛
            eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye = np.array([x + eyes_sorted[0][0] + eyes_sorted[0][2]/2,
                                     y + eyes_sorted[0][1] + eyes_sorted[0][3]/2])
                right_eye = np.array([x + eyes_sorted[-1][0] + eyes_sorted[-1][2]/2,
                                      y + eyes_sorted[-1][1] + eyes_sorted[-1][3]/2])
            else:
                left_eye = np.array([x + w/3, y + h/3])
                right_eye = np.array([x + 2*w/3, y + h/3])

            nose = np.array([x + w/2, y + h/2])
            left_mouth = np.array([x + w/3, y + 2*h/3])
            right_mouth = np.array([x + 2*w/3, y + 2*h/3])

            lm = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
            print("Warning: Using OpenCV Haar cascade for landmarks — results may be inaccurate.")
            print("Install dlib or mediapipe for better landmark detection.")
            return lm

        except Exception:
            return None


def load_deep3d_model(config):
    """加载 Deep3DFaceReconstruction 模型"""
    return Deep3DProcessor(config)