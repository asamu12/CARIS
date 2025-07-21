import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import open3d as o3d
from pathlib import Path
from typing import Tuple, Optional

from RIS.utils.config import load_cfg_from_cfg_file
from RIS.model import build_segmenter
from RIS.utils.dataset import tokenize


from GraspNet.model.FGC_graspnet import FGC_graspnet
from GraspNet.model.decode import pred_decode
from GraspNet.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from GraspNet.utils.collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup

class vl_model():
    def __init__(self, config_path: str, model_weights: str):
        self.cfg = load_cfg_from_cfg_file(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.load_weights(model_weights)
        self.model.eval()
        self.preprocess = self.get_preprocess()

    def build_model(self) -> torch.nn.Module:
        model, _ = build_segmenter(self.cfg)
        return torch.nn.DataParallel(model).to(self.device)

    def load_weights(self, weight_path: str):
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weight file {weight_path} not found")

        checkpoint = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Successfully loaded weights: {weight_path}")

    def get_preprocess(self):
        return Compose([
            Resize((self.cfg.input_size, self.cfg.input_size)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def process_text(self, sentence: str) -> torch.Tensor:
        return tokenize([sentence], self.cfg.word_len, truncate=True).to(self.device)

    def reverse_transform(self, pred: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
        mat_inv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        h, w = orig_size
        return cv2.warpAffine(
            pred, mat_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=0.
        )

    def forward(self, image_path: str, text: str,
                threshold: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform prediction
        :param image_path: Input image path
        :param text: Text description
        :param threshold: Binarization threshold
        :return: (Original image array, Binarized mask array)
        """
        # Read and preprocess image
        orig_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_img.shape[:2]

        # Preprocess
        img_tensor = self.preprocess(Image.fromarray(orig_img))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        text_tensor = self.process_text(text)

        # Model inference
        with torch.no_grad():
            pred = self.model(img_tensor, text_tensor)
            pred = torch.sigmoid(pred.squeeze())

            # Resize
            if pred.shape[-2:] != (orig_h, orig_w):
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            mask_np = pred.cpu().numpy()
            mask_np = self.reverse_transform(mask_np, (orig_h, orig_w))
            binary_mask = (mask_np > threshold).astype(np.uint8) * 255

        return orig_img, binary_mask

    @staticmethod
    def save_result(mask: np.ndarray, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"mask.png"),
            mask
        )


class grasp_model():
    def __init__(self, args, device, image, mask, text) -> None:
        self.args = args
        self.device = device
        self.img = image
        self.text = text
        self.mask = mask

        # Network parameters
        self.num_view = args.num_view
        self.grasp_checkpoint = args.grasp_checkpoint
        self.output_path = args.output_dir
        self.collision_thresh = args.collision_thresh

    def load_grasp_net(self):
        # Initialize the model
        net = FGC_graspnet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                           cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=True)

        net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.grasp_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"-> Loaded FGC_GraspNet checkpoint {self.grasp_checkpoint} (epoch: {start_epoch})")
        # Set model to evaluation mode
        net.eval()
        return net

    def pc_to_depth(self, pc, camera):
        x, y, z = pc
        xmap = x * camera.fx / z + camera.cx
        ymap = y * camera.fy / z + camera.cy

        return int(xmap), int(ymap)

    def choose_in_mask(self, gg):
        camera = CameraInfo(640.0, 480.0, 592.566, 592.566, 319.132, 246.937, 1000)
        gg_new = GraspGroup()
        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            if translation[-1] != 0:
                xmap, ymap = self.pc_to_depth(translation, camera)
                # print(xmap, ymap, self.mask[ymap, xmap])

                if self.mask[ymap, xmap]:
                    gg_new.add(grasp)
        return gg_new

    def get_and_process_data(self, depth):
        # Load data
        color = np.array(Image.fromarray(self.img), dtype=np.float32) / 255.0

        # Use predefined camera parameters
        camera = CameraInfo(640.0, 480.0, 592.566, 592.566, 319.132, 246.937, 1000)

        # Generate point cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # Use the provided mask directly (ensure it is a boolean type)
        valid_mask = self.mask.astype(bool) & (depth > 0)  # Combine object mask and valid depth region
        # Extract valid point cloud
        cloud_masked = cloud[valid_mask]
        color_masked = color[valid_mask]

        # Point cloud sampling
        if len(cloud_masked) >= self.args.num_point:
            idxs = np.random.choice(len(cloud_masked), self.args.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.args.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # Convert to Open3D format
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))

        # Construct network input
        end_points = dict()
        cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device)
        end_points['point_clouds'] = cloud_sampled_tensor
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)

        return gg_array, gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.args.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud):
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        return gg

    def get_top_gg(self, gg):
        if gg.translations.shape[0] == 0:
            return None, None, None
        xyz = gg.translations[0]
        rot = gg.rotation_matrices[0]
        dep = gg.depths[0]
        return xyz, rot, dep

    def check_grasp(self, gg):
        gg_top_down = GraspGroup()
        scores = []

        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            z = translation[2]
            score = grasp.score

            # Target vector for top-down grasp
            target_vector = np.array([0, 0, 1])

            # Grasp approach vector
            grasp_vector = rot[:, 2]  # Assuming the grasp approach vector is the z-axis of the rotation matrix

            # Calculate the angle between the grasp vector and the target vector
            angle = np.arccos(np.clip(np.dot(grasp_vector, target_vector), -1.0, 1.0))

            # Select top-down grasp with a Z value and within 60 degrees (π/3 radians)
            if angle <= np.pi / 4 and z > 0.03:
                gg_top_down.add(grasp)
                scores.append(score)

        if len(scores) == 0:
            return GraspGroup()  # Return an empty GraspGroup if no suitable grasps found

        # Normalize scores and select the best grasps
        ref_value = np.max(scores)
        ref_min = np.min(scores)
        scores = [x - ref_min for x in scores]

        factor = 0.4
        if np.max(scores) > ref_value * factor:
            print('Select top-down grasps')
            return gg_top_down
        else:
            print('No suitable grasp found')
            return GraspGroup()

    def forward(self, depth):
        grasp_net = self.load_grasp_net()
        end_points, cloud = self.get_and_process_data(depth)
        # Generate initial grasp predictions
        gg_array, gg = self.get_grasps(grasp_net, end_points)

        # Regular processing flow
        gg = self.choose_in_mask(gg)  # Filter grasps within the mask area
        gg = self.collision_detection(gg, np.array(cloud.points))  # Collision detection
        # gg = self.check_grasp(gg)
        gg.sort_by_score()  # Sort by confidence score
        gg_array = gg.grasp_group_array

        gg = gg[:5]
        gg = self.vis_grasps(gg, cloud)

        # Save results
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        np.save(f'{self.output_path}/gg.npy', gg_array)
        o3d.io.write_point_cloud(f'{self.output_path}/cloud.ply', cloud)

        xyz, rot, dep = self.get_top_gg(gg)
        return xyz, rot, dep
