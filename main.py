import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from engine import vl_model, grasp_model


def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Language Guided Grasp Detection System")
    parser.add_argument('--img', type=str, default='xxx.png', help='RGB image path')
    parser.add_argument('--depth', type=str, default='xxx.png', help='Depth map path')
    parser.add_argument('--text', type=str, default='language instructions', help='Natural language instruction text')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Result output directory')

    # Vision-language model parameters
    parser.add_argument('--vl_config', type=str,
                        default='xxx.yaml',
                        help='Vision-language model configuration file path')
    parser.add_argument('--vl_weights', type=str,
                        default='xxx.pth',
                        help='Vision-language model weights path')

    # Grasp detection model parameters
    parser.add_argument('--grasp_checkpoint', type=str,
                        default='logs/checkpoint_fgc.tar',
                        help='Grasp detection model checkpoint path')
    parser.add_argument('--num_point', type=int, default=12000, help='Point cloud sampling number')
    parser.add_argument('--num_view', type=int, default=300, help='Viewpoint sampling number')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Collision detection voxel size')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision detection threshold')

    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vl_net = vl_model(
        config_path=args.vl_config,
        model_weights=args.vl_weights
    )
    img, mask = vl_net.forward(args.img, args.text)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    vl_net.save_result(mask, args.output_dir)
    print(f"Segmentation results saved to directory: {args.output_dir}")

    # ==================== Grasp Detection ========================
    depth = np.array(Image.open(args.depth)).astype(np.float32)
    mask_bool = (mask > 128).astype(bool)
    grasp_net = grasp_model(
        args=args,
        device=device,
        image=img,
        mask=mask_bool,
        text=args.text
    )
    xyz, rotation, grasp_depth = grasp_net.forward(depth)
    return xyz, rotation, grasp_depth

if __name__ == "__main__":
    args = parse_args()
    main(args)