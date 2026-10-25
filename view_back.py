from function import *
import cv2
import sys
import os
import csv
import glob
import time
import threading
import queue
import pandas as pd
import numpy as np
import torch.nn.functional as F


# ============================================================
# 定数
# ============================================================
ROUND_NUM = 21
RESIZE_SCALE = 1.0
STATUS_THRESHOLD = 10
POINT_HIP_WINDOW = 10
CAUTION_THRESHOLD = 3 # バーベルの傾き注意する閾値
WARNING_THRESHOLD = 5 # バーベルの傾き警告する閾値
CHECK_CAMERA_VIEW_TILT_THRESHOLD = 7.5 # 画角の異常を判定する際の閾値：肩・腰・膝・踵の傾きの合計


# フラグ
camera_view_ok_flg = False


"""ニーイン推論モデル 指定"""
####################
MODEL_PATH = '../JudgeKneeValgus/JudgeKneeValgusModel.pth'
CLASS_NAMES = ['knee_valgus', 'not_knee_valgus']
INPUT_LENGTH = 150
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = JudgeKneeValgusOneDCNN(in_channels=18, num_classes=len(CLASS_NAMES), input_length=INPUT_LENGTH)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
####################

print("CUDA available:", torch.cuda.is_available())

BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30, 31, 32]

POSE_LANDMARKS = {
    "nose": 0,              # 鼻

    # 左側のキーポイント
    "left_eye_inner": 1,    # 左目（内側）
    "left_eye": 2,          # 左目
    "left_eye_outer": 3,    # 左目（外側）
    "left_ear": 7,          # 左耳
    "left_mouth": 9,        # 口（左側）
    "left_shoulder": 11,    # 左肩
    "left_elbow": 13,       # 左肘
    "left_wrist": 15,       # 左手首
    "left_pinky": 17,       # 左手の小指
    "left_index": 19,       # 左手の人差し指
    "left_thumb": 21,       # 左手の親指
    "left_hip": 23,         # 左股関節
    "left_knee": 25,        # 左膝
    "left_ankle": 27,       # 左足首
    "left_heel": 29,        # 左かかと
    "left_foot_index": 31,  # 左足の人差し指（つま先）

    # 右側のキーポイント
    "right_eye_inner": 4,   # 右目（内側）
    "right_eye": 5,         # 右目
    "right_eye_outer": 6,   # 右目（外側）
    "right_ear": 8,         # 右耳
    "right_mouth": 10,      # 口（右側）
    "right_shoulder": 12,   # 右肩
    "right_elbow": 14,      # 右肘
    "right_wrist": 16,      # 右手首
    "right_pinky": 18,      # 右手の小指
    "right_index": 20,      # 右手の人差し指
    "right_thumb": 22,      # 右手の親指
    "right_hip": 24,        # 右股関節
    "right_knee": 26,       # 右膝
    "right_ankle": 28,      # 右足首
    "right_heel": 30,       # 右かかと
    "right_foot_index": 32, # 右足の人差し指（つま先）

}

BODY_CONNECTIONS = [
    (12, 24), (12, 14), (14, 16), (24, 26), (26, 30), (30, 32),
    (11, 23), (11, 13), (13, 15), (23, 25), (25, 29), (29, 31),
]

REVIEW_COMMENTS = {
    "knee_valgus": "・臀部を下げる際、膝が内側に入っています。つま先と同じ方向に膝を曲げるようにしましょう\n",
    "caution_barbell_tilt": "・身体が左右でやや傾いています。バーベルが水平となるように注意しましょう\n",
    "warning_barbell_tilt": "・身体が左右で非常に傾いています。危険なのでバーベルが水平となるように意識してください\n",
    "stance_width_too_wide": "・足幅が広すぎるので、腰幅から肩幅程度に開きましょう\n",
    "stance_width_too_narrow": "・足幅が狭すぎるので、腰幅から肩幅程度に開きましょう\n",
}

# ------------------------------------------------------------
# 画面表示テキスト（HUD）の座標・色を一元管理
# key: 呼び出し側で使う名前 / value: (x, y, デフォルト色[BGR])
# 座標を変えたい・新しいラベルを増やしたい場合はここだけ編集すればよい
# ------------------------------------------------------------
COLOR_GREEN = [100, 255, 100]
COLOR_MAGENTA = [255, 0, 255]
COLOR_YELLOW = [0, 255, 255]
COLOR_RED = [0, 0, 255]

HUD_LAYOUT = {
    "status":                  (30, 30,  COLOR_GREEN),    # down/up/stay ステータス
    "reps":                    (30, 60,  COLOR_GREEN),    # rep回数
    "recording":               (30, 90,  COLOR_MAGENTA),  # rep記録中フラグ("now")
    "barbell_tilt":            (30, 120, COLOR_YELLOW),   # バーベル傾き警告
    "knee_valgus_result":      (30, 180, COLOR_RED),      # rep終了後のニーイン判定結果
    "shoulder_tilt":           (350, 60,  COLOR_GREEN),   # 肩の傾き
    "hip_tilt":                (350, 90,  COLOR_GREEN),   # 腰の傾き
    "knee_tilt":               (350, 120, COLOR_GREEN),   # 膝の傾き
    "heel_tilt":               (350, 150, COLOR_GREEN),   # 踵の傾き
    "stance_width_ok":         (30, 210, COLOR_GREEN), # 足幅ok
    "stance_width_too_wide":   (30, 210, COLOR_RED),    # 足幅開きすぎ
    "stance_width_too_narrow": (30, 210, COLOR_RED),  # 足幅狭すぎ
}


def draw_label(frame, key: str, text: str, color=None):
    """HUD_LAYOUTに登録された座標にテキストを描画する。
    座標はここでは指定せず、HUD_LAYOUTのキーだけで呼び出す。
    色を一時的に変えたい場合はcolor引数で上書き可能。
    """
    x, y, default_color = HUD_LAYOUT[key]
    draw_text(frame, x, y, text, color if color is not None else default_color)


# ============================================================
# CSV書き込みヘルパー（バッファリング）
# ============================================================
class CsvBuffer:
    """1repのデータをメモリに蓄積し、rep終了時にまとめて書き込む。"""
    def __init__(self):
        self._leg_rows: list = []

    def append_leg(self, *args):
        self._leg_rows.append(args)

    def flush(self, leg_path: str, row_num: int = 150):
        self._write_csv(leg_path, self._leg_rows, num_cols=18, row_num=row_num)
        self._leg_rows = []

    @staticmethod
    def _write_csv(path: str, rows: list, num_cols: int, row_num: int):
        """行数をrow_numに固定してゼロ埋めで書き込む。"""
        padded = list(rows)
        while len(padded) < row_num:
            padded.append([0] * num_cols)
        padded = padded[:row_num]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(padded)


# ============================================================
# ユーティリティクラス
# ============================================================
class Points:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Points(self.x - other.x, self.y - other.y, self.z - other.z)


class AsyncImageWriter:
    """画像書き込みを別スレッドで非同期に行うクラス"""
    def __init__(self, num_workers: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(num_workers)
        ]
        for w in self._workers:
            w.start()

    def write(self, path: str, img):
        try:
            self._queue.put_nowait((path, img))
        except queue.Full:
            cv2.imwrite(path, img)

    def _worker(self):
        while True:
            path, img = self._queue.get()
            if path is None:
                break
            cv2.imwrite(path, img)
            self._queue.task_done()

    def join(self):
        self._queue.join()

    def shutdown(self):
        for _ in self._workers:
            self._queue.put((None, None))
        for w in self._workers:
            w.join()


class RepState:
    """1repごとの状態を管理するクラス"""
    def __init__(self):
        self.count = 0
        self.frame_count = 0
        self.is_recording = False
        self.review_flg = {k: False for k in REVIEW_COMMENTS}
        self.csv_buf = CsvBuffer()

    def start_rep(self):
        self.count += 1
        self.frame_count = 0
        self.is_recording = True

    def end_rep(self):
        self.is_recording = False

    def tick(self):
        self.frame_count += 1


# ============================================================
# メイン処理のヘルパー関数
# ============================================================
def resolve_video_path(video_path: str) -> str:
    if os.path.isfile(video_path):
        return video_path
    raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")


def draw_landmarks(frame, landmarks, frame_shape):
    """全身の骨格を描画する"""
    h, w = frame_shape[:2]
    for i in BODY_INDICES:
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start_idx, end_idx in BODY_CONNECTIONS:
        s, e = landmarks[start_idx], landmarks[end_idx]
        x1, y1 = int(s.x * w), int(s.y * h)
        x2, y2 = int(e.x * w), int(e.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def get_point(landmarks, key: str, frame_shape) -> Points:
    h, w = frame_shape[:2]
    lm = landmarks[POSE_LANDMARKS[key]]
    return Points(int(lm.x * w), int(lm.y * h), lm.z)


def determine_status(hip_deque: deque, threshold: int) -> str:
    """腰のy座標の変動量からステータスを返す"""
    delta = hip_deque[-1] - hip_deque[0]
    if delta >= threshold:
        return "down"
    elif delta <= -threshold:
        return "up"
    return "stay"


def get_tilt_from_two_points(point1: Points, point2: Points) -> float:
    rad_tilt = math.atan((point1.y - point2.y) / (point1.x - point2.x))
    deg_tilt = round(abs(math.degrees(rad_tilt)), 2) # 小数第二位で丸めた傾き

    return deg_tilt


def check_camera_view(frame, landmarks) -> bool:
    """肩・腰・膝・踵の傾きの計算及び描画 これらの情報から画角の異常を判定"""
    shoulder_tilt = get_tilt_from_two_points(landmarks[POSE_LANDMARKS["left_shoulder"]], landmarks[POSE_LANDMARKS["right_shoulder"]])
    draw_label(frame, "shoulder_tilt", f"shoulder_tilt:{shoulder_tilt}")
    
    hip_tilt = get_tilt_from_two_points(landmarks[POSE_LANDMARKS["left_hip"]], landmarks[POSE_LANDMARKS["right_hip"]])
    draw_label(frame, "hip_tilt", f"hip_tilt:{hip_tilt}")
    
    knee_tilt = get_tilt_from_two_points(landmarks[POSE_LANDMARKS["left_knee"]], landmarks[POSE_LANDMARKS["right_knee"]])
    draw_label(frame, "knee_tilt", f"knee_tilt:{knee_tilt}")
    
    heel_tilt = get_tilt_from_two_points(landmarks[POSE_LANDMARKS["left_heel"]], landmarks[POSE_LANDMARKS["right_heel"]])
    draw_label(frame, "heel_tilt", f"heel_tilt:{heel_tilt}")
    
    sum_tilt = (shoulder_tilt + hip_tilt + knee_tilt + heel_tilt)
    
    if sum_tilt < CHECK_CAMERA_VIEW_TILT_THRESHOLD:
        return True
    else:
        return False


def ensure_rep_dirs(base: str, mov_name: str, rep_count: int) -> dict:
    """repごとの出力ディレクトリを一度だけ作成し、パスを返す"""
    dirs = {
        "ori_frame": os.path.join(base, f"ori_frame/{rep_count}"),
        "output":    os.path.join(base, f"output_{mov_name}/{rep_count}"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def detect_barbell_tilt(left_wrist: Points, right_wrist: Points) -> tuple[str,float]:
    """両手首の傾きをバーベルの傾きとして、3～5°なら注意、5°以上なら警告を返す"""
    rad_barbell_tilt = math.atan2(
        (right_wrist.y - left_wrist.y),
        (right_wrist.x - left_wrist.x)
    )
    deg_barbell_tilt = round(abs(math.degrees(rad_barbell_tilt)),2)


    if deg_barbell_tilt < CAUTION_THRESHOLD:
        barbell_tilt_class = "true"
    elif deg_barbell_tilt < WARNING_THRESHOLD:
        barbell_tilt_class = "caution"
    else:
        barbell_tilt_class = "warning"
    return barbell_tilt_class, deg_barbell_tilt


def judge_stance_width(left_elbow: Points, left_shoulder: Points, left_hip: Points, left_heel: Points, right_elbow: Points, right_shoulder: Points, right_hip: Points, right_heel: Points) -> str:
    """肩・腰・かかとの座標関係から足の開き具合の良否判定 かかとが肩より外側にあるまたはかかとが腰より内側にあれば警告"""
    left_shoulder_width = int((left_shoulder.x + left_elbow.x) / 2)
    right_shoulder_width = int((right_shoulder.x + right_elbow.x) / 2)
    left_hip_width = int((left_shoulder.x + left_hip.x) / 2)
    right_hip_width = int((right_shoulder.x + right_hip.x) / 2)
    if left_heel.x < left_shoulder_width and right_shoulder_width < right_heel.x:
        judge = "too_wide"
    elif left_hip_width < left_heel.x and right_heel.x < right_hip_width:
        judge = "too_narrow"
    else:
        judge = "none"
    return judge


def infer_csv(file_path):
    data = pd.read_csv(file_path, header=None).values.astype(np.float32)
    tensor = torch.tensor(data).T.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)   # 確率を計算
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
    return CLASS_NAMES[predicted_class], confidence


def inference_knee_valgus(input_csv_path):
    """input_csv_pathにマッチするCSVを推論し、(ファイルパス, 予測クラス, 確率)のリストを返す"""
    csv_files = glob.glob(input_csv_path)
    if not csv_files:
        print("inference_data にCSVが見つかりません。")
        return []

    results = []
    for file in csv_files:
        pred, conf = infer_csv(file)
        results.append((file, pred, conf))
    return results


# ============================================================
# メイン処理
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python view_back.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = resolve_video_path(sys.argv[1])
    mov_name = os.path.splitext(os.path.basename(video_path))[0]

    start_time = time.time()

    # 第2引数で出力先フォルダが渡された場合はそれを使う（「両方」モード用）。省略時は従来どおり
    if len(sys.argv) >= 3:
        base_path = sys.argv[2]
    else:
        base_path = f"../result/{ROUND_NUM}/{mov_name}"
    save_review_path = os.path.join(base_path, "review.txt")
    save_knee_valgus_csv_path = os.path.join(base_path, f"{mov_name}_knee_valgus_inference.csv")
    os.makedirs(base_path, exist_ok=True)
    csv_dir = os.path.join(base_path, "csvfile")
    os.makedirs(csv_dir, exist_ok=True)

    knee_valgus_inference_log: list = []

    # MediaPipe 初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7
    )

    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 動画を開けませんでした。")
        sys.exit(1)

    # 非同期書き込みワーカー起動
    writer = AsyncImageWriter(num_workers=2)

    # 状態管理
    rep_state = RepState()
    status = "none"
    before_status = "none"
    recent_status = "none"
    point_hip_10frame: deque = deque(maxlen=POINT_HIP_WINDOW)
    frame_count = 0
    rep_dirs: dict = {}
    between_rep_active = False
    between_rep_dir = ""
    between_rep_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("動画の読み込み完了。")
            break

        # 縦向き動画の補正
        h, w = frame.shape[:2]
        if h < w:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = frame.shape[:2]

        if RESIZE_SCALE != 1.0:
            frame = cv2.resize(frame, (int(w * RESIZE_SCALE), int(h * RESIZE_SCALE)))

        original_frame = frame.copy()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result_pose = pose.process(frame_rgb)
        if not result_pose.pose_landmarks:
            frame_count += 1
            continue

        landmarks = result_pose.pose_landmarks.landmark

        draw_landmarks(frame, landmarks, frame.shape)

        pt = {key: get_point(landmarks, key, frame.shape) for key in POSE_LANDMARKS}

        point_hip_10frame.append(pt["right_hip"].y)

        if frame_count >= POINT_HIP_WINDOW:
            status = determine_status(point_hip_10frame, STATUS_THRESHOLD)
            draw_label(frame, "status", status)

        before_status = recent_status
        recent_status = status

        # rep開始検知
        if before_status == "stay" and recent_status == "down":
            if not rep_state.is_recording:
                rep_state.start_rep()
                rep_dirs = ensure_rep_dirs(base_path, mov_name, rep_state.count)
                between_rep_active = False

        draw_label(frame, "reps", f"reps:{rep_state.count}")

        # rep中の処理
        if rep_state.is_recording:
            rep_state.tick()
            draw_label(frame, "recording", "now")

            # 始めの足幅の良否判定
            if rep_state.frame_count == 1:
                """画角の異常判定"""
                camera_view_ok_flg = check_camera_view(frame, landmarks)
                
                """足幅の良否判定"""
                if judge_stance_width(pt["left_elbow"], pt["left_shoulder"], pt["left_hip"], pt["left_heel"], pt["right_elbow"], pt["right_shoulder"], pt["right_hip"], pt["right_heel"]) == "too_wide":
                    rep_state.review_flg["stance_width_too_wide"] = True
                    draw_label(frame, "stance_width_too_wide", "stance_width_too_wide")
                elif judge_stance_width(pt["left_elbow"], pt["left_shoulder"], pt["left_hip"], pt["left_heel"], pt["right_elbow"], pt["right_shoulder"], pt["right_hip"], pt["right_heel"]) == "too_narrow":
                    rep_state.review_flg["stance_width_too_narrow"] = True
                    draw_label(frame, "stance_width_too_narrow", "stance_width_too_narrow")
                else:
                    draw_label(frame, "stance_width_ok", "stance_width_ok")

            barbell_tilt_class, barbell_tilt = detect_barbell_tilt(pt["left_wrist"], pt["right_wrist"])
            if barbell_tilt_class == "true":
                draw_label(frame, "barbell_tilt", f"barbell_not_tilt:{barbell_tilt}", color=COLOR_GREEN)
            elif barbell_tilt_class == "caution":
                rep_state.review_flg["caution_barbell_tilt"] = True
                draw_label(frame, "barbell_tilt", f"slightly_tilt:{barbell_tilt}", color=COLOR_YELLOW)
            elif barbell_tilt_class == "warning":
                rep_state.review_flg["warning_barbell_tilt"] = True
                draw_label(frame, "barbell_tilt", f"heavily_tilt:{barbell_tilt}", color=COLOR_RED)

            # 6部位 × xyz の18列をバッファに追記
            rep_state.csv_buf.append_leg(
                pt["left_knee"].x, pt["left_knee"].y, round(pt["left_knee"].z, 2),
                pt["left_heel"].x, pt["left_heel"].y, round(pt["left_heel"].z, 2),
                pt["left_foot_index"].x, pt["left_foot_index"].y, round(pt["left_foot_index"].z, 2),
                pt["right_knee"].x, pt["right_knee"].y, round(pt["right_knee"].z, 2),
                pt["right_heel"].x, pt["right_heel"].y, round(pt["right_heel"].z, 2),
                pt["right_foot_index"].x, pt["right_foot_index"].y, round(pt["right_foot_index"].z, 2),
            )

            fn = rep_state.frame_count
            writer.write(os.path.join(rep_dirs["ori_frame"], f"{fn}.png"), original_frame)
            writer.write(os.path.join(rep_dirs["output"],    f"{fn}.png"), frame)

        # rep終了検知
        if before_status == "up" and recent_status == "stay" and rep_state.is_recording:
            rep_state.end_rep()

            # rep終了後～次rep開始前のフレーム保存先を準備
            between_rep_dir = os.path.join(base_path, "between_rep", f"between_rep_{rep_state.count}")
            os.makedirs(between_rep_dir, exist_ok=True)
            between_rep_frame_count = 0
            between_rep_active = True

            leg_csv_path = os.path.join(
                csv_dir,
                f"{mov_name}_{rep_state.count}.csv"
            )
            rep_state.csv_buf.flush(leg_csv_path)

            # CNNモデルによるニーイン推論
            infer_results = inference_knee_valgus(leg_csv_path)
            if infer_results:
                _, pred, conf = infer_results[0]
                print(f"[Rep {rep_state.count}] ニーイン推論: {pred} (予測確率: {conf:.4f})")
                knee_valgus_inference_log.append((rep_state.count, pred, conf))
                if pred == "knee_valgus":
                    rep_state.review_flg["knee_valgus"] = True
                elif pred == "not_knee_valgus":
                    rep_state.review_flg["knee_valgus"] = False

        if rep_state.is_recording == False and rep_state.count >= 1:
            if rep_state.review_flg["knee_valgus"] == True:
                draw_label(frame, "knee_valgus_result", f"knee_valgus conf{conf:.4f}", color=COLOR_RED)
            elif rep_state.review_flg["knee_valgus"] == False:
                draw_label(frame, "knee_valgus_result", f"not_knee_valgus conf{conf:.4f}", color=[100, 255, 100])

        # rep間（rep終了後～次rep開始前 or 動画終了まで）の処理後フレームを1枚ずつ保存
        if between_rep_active:
            writer.write(
                os.path.join(between_rep_dir, f"{between_rep_frame_count}.png"),
                frame
            )
            between_rep_frame_count += 1

        cv2.imshow("view back", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 = ESCキー
            break

    # 非同期書き込み完了待ち
    writer.join()
    writer.shutdown()

    # 重要な警告のみを修正コメントに出力
    if rep_state.review_flg["warning_barbell_tilt"]:
        rep_state.review_flg["caution_barbell_tilt"] = False

    # レビューコメント書き込み
    lines = [
        REVIEW_COMMENTS[key]
        for key in rep_state.review_flg
        if rep_state.review_flg[key]
    ]
    with open(save_review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # rep毎のニーイン推論結果をCSVに書き出す
    with open(save_knee_valgus_csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["rep", "pred", "conf"])
        for rep_num, pred, conf in knee_valgus_inference_log:
            writer_csv.writerow([rep_num, pred, f"{conf:.4f}"])

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"処理時間: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    main()