from tkinter import Tk, Label, Button, StringVar, Frame, filedialog, Radiobutton, IntVar
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
import os
import shutil
import cv2
from PIL import Image, ImageTk
import sys

# グローバル変数
video_label = None   # 動画表示用Label
video_cap = None     # OpenCV VideoCaptureオブジェクト
playing = False      # 動画再生フラグ
download_button = None
current_video_path = None

# 選択モード管理
selected_mode = None          # 'back' / 'side' / 'both'
back_file_path = None         # 後視点の選択済みファイルパス
side_file_path = None         # 横視点の選択済みファイルパス
start_button = None           # 処理開始ボタン
frame_container = None        # ドロップエリアのコンテナ
mode_frame = None             # モード選択フレーム

# ===== 動画処理関係 =====

def run_view_script(script_name, video_path):
    subprocess.run(['python', script_name, video_path])

def process_video(mode, file_path):
    """指定モードで動画を処理し、結果を表示・DLボタン追加"""
    if mode == 'back':
        run_view_script('view_back.py', file_path)
        output_path = '../output_back.mp4'
    elif mode == 'side':
        run_view_script('view_side.py', file_path)
        output_path = '../output_side.mp4'
    else:
        return

    if os.path.exists(output_path):
        show_video(output_path)
        enable_download(output_path)
    else:
        print(f"処理後の動画が見つかりません: {output_path}")

def on_start_processing():
    """処理開始ボタン押下時の処理"""
    global selected_mode, back_file_path, side_file_path

    if selected_mode == 'back':
        process_video('back', back_file_path)

    elif selected_mode == 'side':
        process_video('side', side_file_path)

    elif selected_mode == 'both':
        # 後視点 → 横視点の順に処理
        process_video('back', back_file_path)
        process_video('side', side_file_path)

def update_start_button():
    """選択状況に応じて処理開始ボタンの有効/無効を切り替え"""
    global start_button, selected_mode, back_file_path, side_file_path

    if start_button is None:
        return

    ready = False
    if selected_mode == 'back' and back_file_path:
        ready = True
    elif selected_mode == 'side' and side_file_path:
        ready = True
    elif selected_mode == 'both' and back_file_path and side_file_path:
        ready = True

    if ready:
        start_button.config(state='normal', bg="#4CAF50", fg="white")
    else:
        start_button.config(state='disabled', bg="#cccccc", fg="#666666")

def handle_file_selected(file_path, mode, label_var):
    """ファイルが選択されたときの共通処理（処理はまだ実行しない）"""
    global back_file_path, side_file_path

    if os.path.isfile(file_path) and file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        label_var.set(f"✔ {os.path.basename(file_path)}")
        if mode == 'back':
            back_file_path = file_path
        elif mode == 'side':
            side_file_path = file_path
        update_start_button()
    else:
        label_var.set("※ 対応していないファイルです")

def handle_drop(event, mode, label_var):
    file_path = event.data.strip('{}')
    handle_file_selected(file_path, mode, label_var)

def handle_click(mode, label_var):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if file_path:
        handle_file_selected(file_path, mode, label_var)

# ===== 動画再生・保存 =====

def save_video_dialog(video_path):
    save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
    if save_path:
        shutil.copy(video_path, save_path)

def show_video(video_path):
    global video_cap, video_label, playing

    playing = False
    if video_cap:
        video_cap.release()
        video_cap = None

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("動画を開けませんでした。")
        return

    if not video_label:
        global root
        video_label = Label(root)
        video_label.pack(pady=20)

    playing = True
    play_video()
    video_label.bind("<Button-1>", lambda e: save_video_dialog(video_path))

def play_video():
    global video_cap, video_label, playing

    if not playing or not video_cap:
        return

    ret, frame = video_cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(30, play_video)
    else:
        playing = False
        video_cap.release()
        video_cap = None

def enable_download(video_path):
    global download_button

    if download_button:
        download_button.destroy()

    def save_file():
        save_video_dialog(video_path)

    download_button = Button(root, text="処理後動画を保存", command=save_file, bg="#90ee90", font=("Arial", 12))
    download_button.pack(pady=10)

# ===== モード選択→ドロップエリア構築 =====

def create_drop_frame(parent, text, bg_color, mode, col):
    frame = Frame(parent, width=400, height=150, bg=bg_color,
                  highlightbackground="black", highlightthickness=2)
    frame.grid(row=0, column=col, padx=40)
    frame.grid_propagate(False)

    label_text = StringVar(value=text)
    label = Label(frame, textvariable=label_text, font=("Arial", 14),
                  bg=bg_color, justify="center", wraplength=360)
    label.pack(expand=True, fill="both")

    frame.drop_target_register(DND_FILES)
    frame.dnd_bind('<<Drop>>', lambda e: handle_drop(e, mode, label_text))
    label.bind('<Button-1>', lambda e: handle_click(mode, label_text))

    return frame

def build_drop_area():
    """選択されたモードに応じてドロップエリアを構築する"""
    global frame_container, selected_mode, start_button
    global back_file_path, side_file_path

    # リセット
    back_file_path = None
    side_file_path = None

    # 既存のコンテナがあれば破棄
    if frame_container:
        frame_container.destroy()

    frame_container = Frame(root, bg="#f0f0f0")
    frame_container.pack(pady=10)

    if selected_mode == 'back':
        create_drop_frame(frame_container, "後視点（クリック or ドロップ）", "#d0e0ff", "back", 0)

    elif selected_mode == 'side':
        create_drop_frame(frame_container, "横視点（クリック or ドロップ）", "#ffd0d0", "side", 0)

    elif selected_mode == 'both':
        create_drop_frame(frame_container, "後視点（クリック or ドロップ）", "#d0e0ff", "back", 0)
        create_drop_frame(frame_container, "横視点（クリック or ドロップ）", "#ffd0d0", "side", 1)

    # 処理開始ボタン（初期は無効）
    if start_button:
        start_button.destroy()

    start_button = Button(root, text="処理を開始する", command=on_start_processing,
                          font=("Arial", 14, "bold"), state='disabled',
                          bg="#cccccc", fg="#666666", padx=20, pady=8)
    start_button.pack(pady=15)

def on_mode_selected():
    """ラジオボタン変更時にモードを更新してドロップエリアを再構築"""
    global selected_mode
    val = mode_var.get()
    if val == 1:
        selected_mode = 'back'
    elif val == 2:
        selected_mode = 'side'
    elif val == 3:
        selected_mode = 'both'
    build_drop_area()

# ===== GUIの初期設定 =====

root = TkinterDnD.Tk()
root.title("バーベルスクワット評価")
root.geometry("900x300")
root.configure(bg="#f0f0f0")

Label(root, text="バーベルスクワット評価", font=("Arial", 20, "bold"), bg="#f0f0f0").pack(pady=20)

# --- モード選択エリア ---
mode_frame = Frame(root, bg="#f0f0f0")
mode_frame.pack(pady=10)

Label(mode_frame, text="使用する視点を選択してください",
      font=("Arial", 14), bg="#f0f0f0").grid(row=0, column=0, columnspan=3, pady=(0, 10))

mode_var = IntVar(value=0)

Radiobutton(mode_frame, text="後視点のみ", variable=mode_var, value=1,
            font=("Arial", 13), bg="#f0f0f0", command=on_mode_selected).grid(row=1, column=0, padx=30)

Radiobutton(mode_frame, text="横視点のみ", variable=mode_var, value=2,
            font=("Arial", 13), bg="#f0f0f0", command=on_mode_selected).grid(row=1, column=1, padx=30)

Radiobutton(mode_frame, text="両方", variable=mode_var, value=3,
            font=("Arial", 13), bg="#f0f0f0", command=on_mode_selected).grid(row=1, column=2, padx=30)

# frame_container / start_button はモード選択後に動的生成

root.mainloop()