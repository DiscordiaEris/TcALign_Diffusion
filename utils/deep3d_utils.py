import torch
import numpy as np
import os
import cv2
import importlib
from functools import cached_property
from PIL import Image


def _import_deep3d_modules(repo_path):
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(repo_path)
    networks = importlib.import_module("Deep3DFR.models.networks")
    preprocess = importlib.import_module("Deep3DFR.util.preprocess")
    load_mats = importlib.import_module("Deep3DFR.util.load_mats")
    return networks, preprocess.align_img, load_mats.load_lm3d


class Deep3DProcessor:
    """Deep3DFaceReconstruction 处理器"""

    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 通过配置路径加载 Deep3DFR 依赖
        self.networks, self.align_img, self.load_lm3d = _import_deep3d_modules(
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
        model.requires_grad_(False).eval()
        return model

    def process(self, image_tensor):
        """
        处理图像并返回3D参数

        参数:
            image_tensor: 输入图像张量 [1, 3, H, W], 值范围 [-1, 1]

        返回:
            params: 3D参数字典
        """
        pixels = image_tensor.detach().squeeze(0).float().cpu().permute(1, 2, 0)
        rgb = ((pixels.clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
        landmarks = self._detect_landmarks(rgb)
        if landmarks is None:
            raise ValueError("No facial landmarks detected by a trained detector")
        # Deep3DFR alignment expects an upward-positive vertical coordinate.
        landmarks[:, 1] = rgb.shape[0] - 1 - landmarks[:, 1]
        transform, img_aligned_pil, _, _ = self.align_img(
            Image.fromarray(rgb), landmarks, self.lm3d_std
        )
        width, height, scale, center_x, center_y = transform
        resized_width, resized_height = int(width * scale), int(height * scale)
        left = int(resized_width / 2 - 112 + (center_x - width / 2) * scale)
        top = int(resized_height / 2 - 112 + (height / 2 - center_y) * scale)
        alignment = np.array([
            [224 / resized_width, 0, (2 * left + 224) / resized_width - 1],
            [0, 224 / resized_height, (2 * top + 224) / resized_height - 1],
        ], dtype=np.float32)

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
            'alignment': alignment,
        }

        return params

    def reconstruct_aligned(self, images, alignment):
        """Return differentiable coefficients in TcADiff order for aligned portraits."""
        with torch.cuda.amp.autocast(enabled=False):
            images = (images.float().clamp(-1, 1) + 1) / 2
            grid = torch.nn.functional.affine_grid(
                alignment.float(), (len(images), 3, 224, 224), align_corners=False
            )
            aligned = torch.nn.functional.grid_sample(
                images, grid, mode="bicubic", padding_mode="zeros", align_corners=False
            )
            coeff = self.model(aligned.clamp(0, 1))
        return torch.cat((coeff[:, :144], coeff[:, 224:227],
                          coeff[:, 144:224], coeff[:, 227:254]), dim=-1)

    def _detect_landmarks(self, rgb):
        for detect in (self._detect_with_yunet, self._detect_with_dlib,
                       self._detect_with_mediapipe, self._detect_with_mtcnn):
            landmarks = detect(rgb)
            if landmarks is not None:
                return landmarks
        return None

    @cached_property
    def _yunet(self):
        path = self.config.face_detector_path
        if not os.path.isfile(path):
            return None
        return cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold=0.75)

    def detect_yunet_face(self, rgb):
        if self._yunet is None:
            return None
        height, width = rgb.shape[:2]
        self._yunet.setInputSize((width, height))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, faces = self._yunet.detect(bgr)
        if faces is None:
            # Recover dark exposures for detection only. The reconstruction
            # network still receives the original exposure.
            enhanced = cv2.convertScaleAbs(bgr, alpha=2)
            _, faces = self._yunet.detect(enhanced)
        if faces is None:
            return None
        return max(faces, key=lambda row: row[2] * row[3])

    def _detect_with_yunet(self, rgb):
        face = self.detect_yunet_face(rgb)
        if face is None:
            return None
        landmarks = face[4:14].reshape(5, 2).copy()
        landmarks[:2] = sorted(landmarks[:2], key=lambda point: point[0])
        landmarks[3:] = sorted(landmarks[3:], key=lambda point: point[0])
        return landmarks

    @cached_property
    def _dlib_models(self):
        try:
            import dlib
        except ImportError:
            return None
        path = os.path.join(self.config.deep3d_repo_path, "checkpoints",
                            "shape_predictor_68_face_landmarks.dat")
        if not os.path.isfile(path):
            return None
        return dlib.get_frontal_face_detector(), dlib.shape_predictor(path)

    def _detect_with_dlib(self, img):
        if self._dlib_models is None:
            return None
        detector, predictor = self._dlib_models
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = detector(gray, 1)
        if not faces:
            return None
        face = max(faces, key=lambda box: box.width() * box.height())
        shape = predictor(gray, face)
        points = np.array([(point.x, point.y) for point in shape.parts()], dtype=np.float32)
        return np.stack((points[36:42].mean(0), points[42:48].mean(0),
                         points[30], points[48], points[54]))

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
                results = face_mesh.process(img)

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
            pil_img = Image.fromarray(img)
            boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)

            if boxes is None or len(boxes) == 0:
                return None

            # points shape: (5, 2) — left_eye, right_eye, nose, left_mouth, right_mouth
            return points[0].astype(np.float32)

        except Exception:
            return None


def load_deep3d_model(config):
    """加载 Deep3DFaceReconstruction 模型"""
    return Deep3DProcessor(config)
