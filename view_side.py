from function import *
import cv2
import sys
import os
import csv
import time
import threading
import queue
import pandas as pd

# 定数
ROUND_NUM = 18
RESIZE_SCALE = 1.0
STATUS_THRESHOLD = 10       # 腰のy座標変動量（px）でstatus判定するしきい値
HEAD_ANGLE_THRESHOLD = 5    # 頭と身体の傾き差（度）で警告するしきい値
POINT_HIP_WINDOW = 10       # 腰y座標を保持するフレーム数

print("CUDA available:", torch.cuda.is_available())

# 右半身のランドマークインデックス
RIGHT_INDICES = [5, 10, 12, 24, 26, 30, 32]

POSE_LANDMARKS = {
    "right_eye": 5,
    "right_mouth": 10,
    "right_shoulder": 12,
    "right_hip": 24,
    "right_knee": 26,
    "right_heel": 30,
    "right_foot_index": 32,
}

RIGHT_CONNECTIONS = [
    (5, 10),
    (12, 24),
    (24, 26),
    (26, 30),
]

REVIEW_COMMENTS = {
    "deep_depth": "・臀部を下げすぎています。怪我の原因となるので太ももが床と平衡になる程度で止めましょう\n",
    "shallow_depth": "・臀部が下がりきっていません。太ももが床と平行になる程度まで臀部を降ろしましょう\n",
    "head_too_tilt": "・下を向き過ぎです。頭の角度と身体の角度が同じになるようにしましょう\n",
    "head_not_tilt": "・前を向き過ぎです。頭の角度と身体の角度が同じになるようにしましょう\n",
    "head_unstable": "・下を向き過ぎたり前を見過ぎたりしています。頭の角度と身体の角度が同じになるようにしましょう\n",
    "torso_too_upright": "・体幹が直立すぎるので少し前傾しましょう\n",
    "torso_too_forward": "・体幹が前傾過ぎるので少し後傾しましょう\n"
}

# ------------------------------------------------------------
# 画面表示テキスト（HUD）の座標・色を一元管理
# key: 呼び出し側で使う名前 / value: (x, y, デフォルト色[BGR])
# 座標を変えたい・新しいラベルを増やしたい場合はここだけ編集すればよい
# ------------------------------------------------------------
COLOR_GREEN = [100, 255, 100]
COLOR_MAGENTA = [255, 0, 255]
COLOR_CYAN = [255, 255, 0]
COLOR_RED = [0, 0, 255]

HUD_LAYOUT = {
    "status":            (30, 30,  COLOR_GREEN),    # down/up/stay ステータス
    "reps":              (30, 60,  COLOR_GREEN),    # rep回数
    "recording":         (30, 90,  COLOR_MAGENTA),  # rep記録中フラグ("now")
    "theta_head_body":   (30, 120, COLOR_CYAN),     # 頭と身体の角度差
    "head_tilt":         (30, 150, COLOR_RED),     # 頭の向き警告(head_not_tilt/head_too_tilt)
    "depth_result":      (30, 180, COLOR_RED),      # rep終了後のしゃがみ深度判定結果
    "torso_too_upright": (30, 180, COLOR_RED),      # 体幹が直立しすぎ
    "torso_too_forward": (30, 180, COLOR_RED),      # 体幹が前傾しすぎ
    "torso_ok":          (30, 180, COLOR_GREEN),    # 体幹おｋ
}


def draw_label(frame, key: str, text: str, color=None):
    """HUD_LAYOUTに登録された座標にテキストを描画する。
    座標はここでは指定せず、HUD_LAYOUTのキーだけで呼び出す。
    色を一時的に変えたい場合はcolor引数で上書き可能。
    """
    x, y, default_color = HUD_LAYOUT[key]
    draw_text(frame, x, y, text, color if color is not None else default_color)


# ユーティリティクラス
class Points:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Points(self.x - other.x, self.y - other.y)


class AsyncImageWriter:
    """画像書き込みを別スレッドで非同期に行うクラス。
    メインループのI/Oブロッキングを解消する。
    """
    def __init__(self, num_workers: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(num_workers)
        ]
        for w in self._workers:
            w.start()

    def write(self, path: str, img):
        """書き込みキューに追加する（ブロッキングなし）。"""
        try:
            self._queue.put_nowait((path, img))
        except queue.Full:
            # キューが満杯の場合は同期書き込みにフォールバック
            cv2.imwrite(path, img)

    def _worker(self):
        while True:
            path, img = self._queue.get()
            if path is None:
                break
            cv2.imwrite(path, img)
            self._queue.task_done()

    def join(self):
        """残りすべての書き込みが完了するまで待機する。"""
        self._queue.join()

    def shutdown(self):
        """ワーカースレッドを終了させる。"""
        for _ in self._workers:
            self._queue.put((None, None))
        for w in self._workers:
            w.join()


class RepState:
    """1repごとの状態を管理するクラス。"""
    def __init__(self):
        self.count = 0
        self.frame_count = 0
        self.is_recording = False
        self.review_flg = {k: False for k in REVIEW_COMMENTS}

    def start_rep(self):
        self.count += 1
        self.frame_count = 0
        self.is_recording = True

    def end_rep(self):
        self.is_recording = False

    def tick(self):
        self.frame_count += 1


# CSV書き込みヘルパー（バッファリング）
class CsvBuffer:
    """1repのデータをメモリに蓄積し、rep終了時にまとめて書き込む。"""
    def __init__(self):
        self._point_rows: list = []
        self._theta_rows: list = []

    def append_point(self, *args):
        self._point_rows.append(args)

    def append_theta(self, *args):
        self._theta_rows.append(args)

    def flush(self, point_path: str, theta_path: str, row_num: int = 300):
        self._write_csv(point_path, self._point_rows, num_cols=6, row_num=row_num)
        self._write_csv(theta_path, self._theta_rows, num_cols=2, row_num=row_num)
        self._point_rows = []
        self._theta_rows = []

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


# メイン処理
def resolve_video_path(video_path: str) -> str:
    """引数のパスをそのまま使う。存在しない場合はエラー。"""
    if os.path.isfile(video_path):
        return video_path
    raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")


def draw_landmarks(frame, landmarks, frame_shape):
    """右半身の骨格を描画する。"""
    h, w = frame_shape[:2]
    for i in RIGHT_INDICES:
        if i == 32:
            continue
        lm = landmarks[i]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start_idx, end_idx in RIGHT_CONNECTIONS:
        s, e = landmarks[start_idx], landmarks[end_idx]
        x1, y1 = int(s.x * w), int(s.y * h)
        x2, y2 = int(e.x * w), int(e.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def get_point(landmarks, key: str, frame_shape) -> Points:
    h, w = frame_shape[:2]
    lm = landmarks[POSE_LANDMARKS[key]]
    return Points(int(lm.x * w), int(lm.y * h))


def determine_status(hip_deque: deque, threshold: int) -> str:
    """腰のy座標の変動量からステータスを返す。"""
    delta = hip_deque[-1] - hip_deque[0]
    if delta >= threshold:
        return "down"
    elif delta <= -threshold:
        return "up"
    return "stay"


def compute_head_angle(p_eye, p_mouth, p_shoulder, p_hip) -> tuple[float, float, float]:
    """顔ベクトルと身体ベクトルの角度差と各傾きを返す。"""
    p0 = np.array([p_eye.x, p_eye.y])
    p1 = np.array([p_mouth.x, p_mouth.y])
    p2 = np.array([p_shoulder.x, p_shoulder.y])
    p3 = np.array([p_hip.x, p_hip.y])

    vec_face = p0 - p1
    vec_body = p2 - p3

    cos_theta = np.dot(vec_face, vec_body) / (
        np.linalg.norm(vec_face) * np.linalg.norm(vec_body) + 1e-9
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    face_tilt = abs(vec_face[1]) / (abs(vec_face[0]) + 1e-9)
    body_tilt = abs(vec_body[1]) / (abs(vec_body[0]) + 1e-9)

    return theta, face_tilt, body_tilt


def ensure_rep_dirs(base: str, round_num: int, mov_name: str, rep_count: int) -> dict:
    """repごとの出力ディレクトリを一度だけ作成し、パスを返す。"""
    dirs = {
        "ori_frame": os.path.join(base, f"ori_frame/{rep_count}"),
        # "skeleton": os.path.join(base, f"skeleton/{rep_count}"),
        "output": os.path.join(base, f"output_{mov_name}/{rep_count}"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def judge_torso_tilt(hip_point: Points, shoulder_point: Points) -> tuple[str, int]:
    dx = shoulder_point.x - hip_point.x
    dy = shoulder_point.y - hip_point.y

    angle_rad = math.atan2(dx, dy)

    # ラジアンから度に変換
    angle_deg = int(180 - math.degrees(angle_rad))
    print(f"angle = {angle_deg}")

    """35度未満は直立すぎ、45度以上は前傾すぎ"""
    if 45 < angle_deg:
        return "too_forward", angle_deg
    elif angle_deg < 35:
        return "too_upright", angle_deg
    else:
        return "none", angle_deg

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_side.py <video_path>")
        sys.exit(1)

    video_path = resolve_video_path(sys.argv[1])
    mov_name = os.path.splitext(os.path.basename(video_path))[0]
    user_name = os.path.basename(os.path.dirname(video_path))

    start_time = time.time()

    # ディレクトリ初期化
    base_path = f"../result/{ROUND_NUM}/{mov_name}"
    csv_dir = os.path.join(base_path, "csvfile")
    save_review_path = os.path.join(base_path, "review.txt")
    for d in [base_path, csv_dir]:
        os.makedirs(d, exist_ok=True)

    # MediaPipe 初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=2,  # モデルの精度を指定 0: Lite, 1: Full, 2: Heavy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
    csv_buf = CsvBuffer()
    status = "none"
    before_status = "none"
    recent_status = "none"
    point_hip_10frame: deque = deque(maxlen=POINT_HIP_WINDOW)
    frame_count = 0
    rep_dirs: dict = {}
    between_rep_active = False
    between_rep_dir = ""
    between_rep_frame_count = 0
    csv_point_path = ""
    csv_theta_path = ""

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
        # input_pose = contrast_custom_tone_curve(frame_rgb)

        # result_pose = pose.process(input_pose)
        result_pose = pose.process(frame_rgb)
        if not result_pose.pose_landmarks:
            frame_count += 1
            continue

        landmarks = result_pose.pose_landmarks.landmark

        # 骨格描画
        draw_landmarks(frame, landmarks, frame.shape)
        frame_skeleton = frame.copy()

        # 各部位の座標取得
        pt = {key: get_point(landmarks, key, frame.shape) for key in POSE_LANDMARKS}

        point_hip_10frame.append(pt["right_hip"].y)

        # ステータス判定（10フレーム以降）
        if frame_count >= POINT_HIP_WINDOW:
            status = determine_status(point_hip_10frame, STATUS_THRESHOLD)
            draw_label(frame, "status", status)

        # ステータス履歴更新
        before_status = recent_status
        recent_status = status

        # rep開始検知
        if before_status == "stay" and recent_status == "down":
            if not rep_state.is_recording:
                rep_state.start_rep()
                rep_dirs = ensure_rep_dirs(base_path, ROUND_NUM, mov_name, rep_state.count)
                csv_point_path = os.path.join(csv_dir, f"{mov_name}point{rep_state.count}.csv")
                csv_theta_path = os.path.join(csv_dir, f"{mov_name}theta{rep_state.count}.csv")
                between_rep_active = False

        draw_label(frame, "reps", f"reps:{rep_state.count}")

        # rep中の処理
        if rep_state.is_recording:
            rep_state.tick()
            draw_label(frame, "recording", "now")

            # 体感前傾角度の良否判定
            if status == "stay" and before_status == "down":
                print("はいったはいった\n")
                torso_tilt_flg, torso_tilt = judge_torso_tilt(pt["right_hip"], pt["right_shoulder"])
                if torso_tilt_flg == "too_upright":
                    print("upright")
                    rep_state.review_flg["torso_too_upright"] = True
                    draw_label(frame, "torso_too_upright", f"torso_too_upright:{torso_tilt}")
                elif torso_tilt_flg == "too_forward":
                    print("forward")
                    rep_state.review_flg["torso_too_forward"] = True
                    draw_label(frame, "torso_too_forward", f"torso_too_forward:{torso_tilt}")
                else:
                    draw_label(frame, "torso_ok", f"torso_ok:{torso_tilt}")
                    print("none")


            # 角度計算
            theta_hip = calculate_theta(pt["right_knee"], pt["right_hip"], pt["right_shoulder"])
            theta_knee = calculate_theta(pt["right_hip"], pt["right_knee"], pt["right_heel"])

            color = [255, 255, 0]
            cv2.line(frame,
                     (pt["right_hip"].x, pt["right_hip"].y),
                     (pt["right_hip"].x + 70, pt["right_hip"].y), color, 2)
            draw_text(frame, pt["right_hip"].x + 75, pt["right_hip"].y + 10,
                      f"theta:{theta_hip:.1f}", color)

            cv2.line(frame,
                     (pt["right_knee"].x, pt["right_knee"].y),
                     (pt["right_knee"].x - 50, pt["right_knee"].y + 30), color, 2)
            draw_text(frame, pt["right_knee"].x - 120, pt["right_knee"].y + 60,
                      f"theta:{theta_knee:.1f}", color)

            # 頭の向き判定
            theta_head_body, face_tilt, body_tilt = compute_head_angle(
                pt["right_eye"], pt["right_mouth"],
                pt["right_shoulder"], pt["right_hip"]
            )
            if theta_head_body >= HEAD_ANGLE_THRESHOLD:
                if face_tilt > body_tilt:
                    rep_state.review_flg["head_not_tilt"] = True
                    draw_label(frame, "head_tilt", "head_not_tilt")
                else:
                    rep_state.review_flg["head_too_tilt"] = True
                    draw_label(frame, "head_tilt", "head_too_tilt")
            draw_label(frame, "theta_head_body", f"theta:{theta_head_body:.1f}")

            # CSVバッファへ追記
            csv_buf.append_point(
                pt["right_shoulder"].x, pt["right_shoulder"].y,
                pt["right_hip"].x, pt["right_hip"].y,
                pt["right_knee"].x, pt["right_knee"].y,
            )
            csv_buf.append_theta(theta_hip, theta_knee)

            # 画像を非同期書き込みキューへ追加
            fn = rep_state.frame_count
            writer.write(os.path.join(rep_dirs["ori_frame"], f"{fn}.png"), original_frame)
            # writer.write(os.path.join(rep_dirs["skeleton"], f"{fn}.png"), frame_skeleton)
            writer.write(os.path.join(rep_dirs["output"], f"{fn}.png"), frame)

        # rep終了検知
        if before_status == "up" and recent_status == "stay" and rep_state.is_recording:
            # 先にCSVを書き込む（推論がファイルを必要とするため）
            csv_buf.flush(csv_point_path, csv_theta_path)

            # CSV書き込み完了後に推論
            pred, conf = one_dimension_cnn(csv_point_path)
            if pred == "shallow":
                rep_state.review_flg["shallow_depth"] = True
                rep_state.review_flg["deep_depth"] = False
            elif pred == "deep":
                rep_state.review_flg["deep_depth"] = True
                rep_state.review_flg["shallow_depth"] = False
            elif pred == "true":
                rep_state.review_flg["shallow_depth"] = False
                rep_state.review_flg["deep_depth"] = False


            rep_state.end_rep()

            # rep終了後～次rep開始前のフレーム保存先を準備
            between_rep_dir = os.path.join(base_path, "between_rep", f"between_rep_{rep_state.count}")
            os.makedirs(between_rep_dir, exist_ok=True)
            between_rep_frame_count = 0
            between_rep_active = True

        if rep_state.is_recording == False and rep_state.count >= 1:
            if rep_state.review_flg["deep_depth"] == True:
                draw_label(frame, "depth_result", f"squat_depth:deep {conf:.4f}")
            elif rep_state.review_flg["shallow_depth"] == True:
                draw_label(frame, "depth_result", f"squat_depth:shallow {conf:.4f}", color=[255, 0, 0])
            elif rep_state.review_flg["deep_depth"] == False and rep_state.review_flg["shallow_depth"] == False:
                draw_label(frame, "depth_result", f"squat_depth:true {conf:.4f}", color=[0, 255, 0])

        # rep間（rep終了後～次rep開始前 or 動画終了まで）の処理後フレームを1枚ずつ保存
        if between_rep_active and between_rep_frame_count == 0:
            between_rep_frame_count += 1
            writer.write(
                os.path.join(between_rep_dir, f"{between_rep_frame_count}.png"),
                frame
            )

        cv2.imshow("view side", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 = ESCキー
            break

    # 非同期書き込み完了待ち
    writer.join()
    writer.shutdown()

    # 顔の向きが下向きすぎと前向きすぎの両方判定された場合の処理　出力コメントを変更
    if rep_state.review_flg["head_not_tilt"] and rep_state.review_flg["head_too_tilt"]:
        rep_state.review_flg["head_not_tilt"] = False
        rep_state.review_flg["head_too_tilt"] = False
        rep_state.review_flg["head_unstable"] = True

    # レビューコメント書き込み
    lines = [
        REVIEW_COMMENTS[key]
        for key in rep_state.review_flg
        if rep_state.review_flg[key]
    ]
    with open(save_review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 後処理
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"処理時間: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    main()