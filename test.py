import cv2
import mmcv
from mmpose.apis import init_pose_model, inference_top_down_pose_model
import time

cap = cv2.VideoCapture("storage/3.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 動画の画面横幅
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 動画の画面縦幅
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 総フレーム数
frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) # フレームレート(fps)
print(height,width,frame_count,frame_rate)
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
writer = cv2.VideoWriter('vis_results/output.mp4', fmt, frame_rate, (width, height)) # ライター作成
# モデル構成ファイルのパスを指定
config_file = "configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w48_animalpose_256x256.py"
# 学習済みの重みファイルのパスを指定
checkpoint_file = "work_dirs/hrnet_w48_animalpose_256x256/epoch_50.pth"
# モデルを初期化
model = init_pose_model(config_file, checkpoint_file, device='cuda:0')

# 動画のフレームを1つずつ読み込み、骨格検知を行い、結果を出力する
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # フレームをモデルに入力し、骨格検知を行う
        pose_results, _ = inference_top_down_pose_model(model, frame)
        # 骨格検知の結果を描画する
        vis_frame = model.show_result(frame, pose_results, show=False)
        # 結果を表示する
        # cv2.imshow('result', vis_frame)
        writer.write(vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

writer.release()
# 終了処理
cap.release()
cv2.destroyAllWindows()