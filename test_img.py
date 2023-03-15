import glob
import cv2
import mmcv
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import time


path = "data/horse_original/test_img/"
files = glob.glob(path+'*')
config_file = "configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w48_animalpose_256x256.py"
checkpoint_file = "work_dirs/hrnet_w48_animalpose_256x256/epoch_50.pth"
model = init_pose_model(config_file, checkpoint_file, device='cuda:0')

for f in files:
    file_name = f.replace(path,"")
    pose_results, _ = inference_top_down_pose_model(model, f)
    vis_frame = model.show_result(f, pose_results, show=False)
    cv2.imwrite(
            f"vis_results/{file_name}",
            vis_frame,
        )