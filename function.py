import math
import cv2
import os
import pandas as pd
import numpy as np
import mediapipe as mp
from collections import deque
import os
import glob
import torch
import torch.nn.functional as F
import sys
# OneDCNN / JudgeKneeValgus パッケージが1つ上のディレクトリにあるため、親ディレクトリを import 検索パスに追加
# （import 文には '../' を書けないので sys.path で対応する。append なのでこのディレクトリ内の function.py が優先される）
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from OneDCNN.OneDCNNmodel import SimpleOneDCNN
from JudgeKneeValgus.JudgeKneeValgusModel import JudgeKneeValgusOneDCNN

"""view_side スクワットの深さ推論モデル　指定"""
####################
MODEL_PATH = '../OneDCNN/onedcnn_model.pth'
CLASS_NAMES = ['deep', 'shallow', 'true']
INPUT_LENGTH = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SimpleOneDCNN(in_channels=6, num_classes=len(CLASS_NAMES), input_length=INPUT_LENGTH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
####################

# print(torch.cuda.is_available())  # TrueならGPU使用可能

def increase_saturation_brightness(
    image,
    sat_scale=1.3,   # 彩度の倍率（1.0が変更なし）
    val_scale=1.2    # 明度の倍率（1.0が変更なし）
):
    # BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # 彩度・明度を倍率で増加
    s = np.clip(s.astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * val_scale, 0, 255).astype(np.uint8)

    # HSVを再構成
    hsv_enhanced = cv2.merge([h, s, v])

    # HSV → BGR
    output = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return output

#　作成したトーンカーブの関数に画像を入力し、コントラスト強調後の画像を出力する
def contrast_custom_tone_curve(image):
    x = np.arange(256)
    y = 300 / (1 + np.exp(-(x - 150) / 80))

    # 白飛び防止のため 0–255 にクリップ
    y = np.clip(y, 0, 255)

    lut = y.astype(np.uint8)
    return cv2.LUT(image, lut)

def enhance_for_pose(image):
    # 明るさ成分（Lチャンネル）に対してのみ処理を行うためLAB色空間に変換
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # clipLimitで強調度合いを調整（2.0〜5.0程度が一般的）
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # 結合して元のBGRに戻す
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def infer_csv(file_path):
    data = pd.read_csv(file_path, header=None).values.astype(np.float32)
    tensor = torch.tensor(data).T.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)   # 確率を計算
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
    return CLASS_NAMES[predicted_class], confidence

def one_dimension_cnn(input_csv_path):
    csv_files = glob.glob(input_csv_path)
    if not csv_files:
        print("inference_data にCSVが見つかりません。")
        return

    print("推論結果:")
    for file in csv_files:
        pred, conf = infer_csv(file)
        print(f"{os.path.basename(file)} → {pred} （確率: {conf:.4f}）")

    return pred, conf
####################

# 求めたいthetaをpoint1(x1, y1)とする．時計周りにpoint2(x2, y2) -> point0(x0, y0)の順
def calculate_theta(point0, point1, point2):
    # Pointsオブジェクト → NumPy配列に変換
    p0 = np.array([point0.x, point0.y])
    p1 = np.array([point1.x, point1.y])
    p2 = np.array([point2.x, point2.y])

    # ベクトルを求める
    v1 = p0 - p1
    v2 = p2 - p1

    # 内積と角度計算
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))
    return theta

# csvファイルにデータを入力
def WritePointsCsv(point_x1, point_y1, point_x2, point_y2, point_x3, point_y3, csv_file_path):
    # 書き込む新しい1行データ
    new_row = [point_x1, point_y1, point_x2, point_y2, point_x3, point_y3]

    row_num = 300

    # CSVが存在し、中身がある場合は読み込む。なければrow_num行ゼロで作成
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        df = pd.read_csv(csv_file_path, header=None)
    else:
        df = pd.DataFrame(np.zeros((row_num, 6)))

    # 最初の空行（ゼロ行）を探す
    for i in range(row_num):
        if (df.iloc[i] == 0).all():
            df.iloc[i] = new_row
            break

    # 保存（row_num行まで）
    df.iloc[:row_num].to_csv(csv_file_path, index=False, header=False)

# csvファイルにデータを入力
def WriteThetaCsv(theta1, theta2, csv_file_path):
    # 新しいデータ（1行分）をリスト形式で用意
    new_data = [theta1, theta2]
    row_num = 300

    # CSVが存在し、中身がある場合は読み込む。なければrow_num行ゼロで作成
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        df = pd.read_csv(csv_file_path, header=None)
    else:
        df = pd.DataFrame(np.zeros((row_num, 2)))

    # 最初の空行（ゼロ行）を探す
    for i in range(row_num):
        if (df.iloc[i] == 0).all():
            df.iloc[i] = new_data
            break
    # 保存
    df.iloc[:row_num].to_csv(csv_file_path, index=False, header=False)

def draw_text(frame, x, y, text, color):
    cv2.putText(
        frame,  # 画像
        text,  # 描画するテキスト
        (x, y),  # 描画位置（x, y）
        cv2.FONT_HERSHEY_SIMPLEX,  # フォント
        1.0,  # 文字サイズ（スケール）
        color,  # 色（BGR）
        2,  # 太さ
        cv2.LINE_AA  # アンチエイリアス
    )

def draw_landmark_circle(frame, landmarks, landmark_indices, radius):
    landmark = landmarks[landmark_indices]
    x = int(landmark.x * frame.shape[1])
    y = int(landmark.y * frame.shape[0])
    cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)