from tkinter import Tk, Label, Button, StringVar, Frame, filedialog, Radiobutton, IntVar
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
import os
import shutil
import cv2
from PIL import Image, ImageTk
import sys

# ===== 出力先の設定 =====
# view_back.py / view_side.py の ROUND_NUM と合わせること（結果は ../result/<ROUND_NUM>/ 以下に出力される）
ROUND_NUM = 23
RESULT_ROOT = os.path.join('..', 'result', str(ROUND_NUM))

# 対応する動画拡張子
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')

# グローバル変数
video_label = None   # 動画表示用Label
video_cap = None     # OpenCV VideoCaptureオブジェクト
playing = False      # 動画再生フラグ
download_button = None
current_video_path = None

# 選択モード管理
selected_mode = None          # 'back' / 'side' / 'both'
back_file_paths = []          # 後視点の選択済みファイルパス（複数）
side_file_paths = []          # 横視点の選択済みファイルパス（複数）
start_button = None           # 処理開始ボタン
frame_container = None        # ドロップエリアのコンテナ
mode_frame = None             # モード選択フレーム
status_label = None           # 進捗・警告表示用Label
status_var = None             # 進捗・警告表示用StringVar

# ===== 動画処理関係 =====

def run_view_script(script_name, video_path, output_dir=None):
    """view_*.py を実行する。output_dir を渡すとその場所に結果を出力させる（省略時は従来の出力先）"""
    cmd = ['python', script_name, video_path]
    if output_dir:
        cmd.append(output_dir)
    return subprocess.run(cmd).returncode

def process_video(mode, file_path, output_dir=None):
    """指定モードで動画を処理し、結果を表示・DLボタン追加"""
    if mode == 'back':
        rc = run_view_script('view_back.py', file_path, output_dir)
        output_path = '../output_back.mp4'
    elif mode == 'side':
        rc = run_view_script('view_side.py', file_path, output_dir)
        output_path = '../output_side.mp4'
    else:
        return 1

    if os.path.exists(output_path):
        show_video(output_path)
        enable_download(output_path)
    else:
        print(f"処理終了")
    return rc

def set_status(text):
    """進捗・警告表示を更新する"""
    if status_var is not None:
        status_var.set(text)
        root.update()

def on_start_processing():
    """処理開始ボタン押下時の処理（選択された動画を順番に1本ずつ処理する）"""
    global selected_mode, back_file_paths, side_file_paths

    # 処理中の二重実行を防ぐ
    start_button.config(state='disabled', bg="#cccccc", fg="#666666")

    try:
        if selected_mode == 'back':
            total = len(back_file_paths)
            for i, path in enumerate(back_file_paths, 1):
                set_status(f"処理中 ({i}/{total}): {os.path.basename(path)}")
                process_video('back', path)

        elif selected_mode == 'side':
            total = len(side_file_paths)
            for i, path in enumerate(side_file_paths, 1):
                set_status(f"処理中 ({i}/{total}): {os.path.basename(path)}")
                process_video('side', path)

        elif selected_mode == 'both':
            # ペアにはせず、選択された動画を1本ずつ処理する（後視点 → 横視点の順）
            #   出力先は「後視点のみ」「横視点のみ」モードと同じ
            jobs = [('back', p) for p in back_file_paths] + [('side', p) for p in side_file_paths]
            total = len(jobs)
            failed = []
            for i, (kind, path) in enumerate(jobs, 1):
                set_status(f"処理中 ({i}/{total}): {os.path.basename(path)}")
                try:
                    if process_video(kind, path) != 0:
                        failed.append(os.path.basename(path))
                except Exception as e:
                    # 1本で例外が出ても次へ進む
                    print(f"{os.path.basename(path)} の処理中にエラー: {e}")
                    failed.append(os.path.basename(path))

            if failed:
                set_status(f"完了（失敗あり: {', '.join(failed)}）")
                return

        set_status("すべての処理が完了しました")
    finally:
        update_start_button()

def is_ready():
    """処理を開始できる状態か"""
    if selected_mode == 'back':
        return len(back_file_paths) > 0
    if selected_mode == 'side':
        return len(side_file_paths) > 0
    if selected_mode == 'both':
        # 本数が違っていても、どちらかに1本以上選ばれていれば開始できる（1本ずつ処理する）
        return len(back_file_paths) > 0 or len(side_file_paths) > 0
    return False

def update_start_button():
    """選択状況に応じて処理開始ボタンの有効/無効を切り替え"""
    global start_button

    if start_button is None:
        return

    if is_ready():
        start_button.config(state='normal', bg="#4CAF50", fg="white")
    else:
        start_button.config(state='disabled', bg="#cccccc", fg="#666666")

def update_count_warning():
    """本数が違っていても1本ずつ処理できるため、案内は出さない（呼び出し側との互換用）"""
    if status_var is not None:
        status_var.set("")

def make_selection_text(paths):
    """ドロップエリアに表示する選択結果の文字列"""
    names = [os.path.basename(p) for p in paths]
    if len(names) == 1:
        return f"✔ {names[0]}"
    shown = names[:3]
    text = f"✔ {len(names)}件選択\n" + "\n".join(shown)
    if len(names) > 3:
        text += "\n…"
    return text

def handle_files_selected(file_paths, mode, label_var):
    """ファイルが選択されたときの共通処理（処理はまだ実行しない）"""
    global back_file_paths, side_file_paths

    valid = [p for p in file_paths
             if os.path.isfile(p) and p.lower().endswith(VIDEO_EXTS)]

    if valid:
        # 名前順に並べて、処理順・「両方」モードでのペアリングを安定させる
        valid.sort(key=lambda p: os.path.basename(p).lower())
        label_var.set(make_selection_text(valid))
        if mode == 'back':
            back_file_paths = valid
        elif mode == 'side':
            side_file_paths = valid
        update_count_warning()
        update_start_button()
    else:
        label_var.set("※ 対応していないファイルです")

def handle_drop(event, mode, label_var):
    # 複数ファイルのドロップに対応（スペースを含むパスは {} で囲まれて渡される）
    file_paths = list(root.tk.splitlist(event.data))
    handle_files_selected(file_paths, mode, label_var)

def handle_click(mode, label_var):
    file_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")])
    if file_paths:
        handle_files_selected(list(file_paths), mode, label_var)

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
    global back_file_paths, side_file_paths
    global status_label, status_var

    # リセット
    back_file_paths = []
    side_file_paths = []

    # 既存のコンテナがあれば破棄
    if frame_container:
        frame_container.destroy()

    frame_container = Frame(root, bg="#f0f0f0")
    frame_container.pack(pady=10)

    if selected_mode == 'back':
        create_drop_frame(frame_container, "後視点（クリック or ドロップ・複数可）", "#d0e0ff", "back", 0)

    elif selected_mode == 'side':
        create_drop_frame(frame_container, "横視点（クリック or ドロップ・複数可）", "#ffd0d0", "side", 0)

    elif selected_mode == 'both':
        create_drop_frame(frame_container, "後視点（クリック or ドロップ・複数可）", "#d0e0ff", "back", 0)
        create_drop_frame(frame_container, "横視点（クリック or ドロップ・複数可）", "#ffd0d0", "side", 1)

    # 処理開始ボタン（初期は無効）
    if start_button:
        start_button.destroy()

    start_button = Button(root, text="処理を開始する", command=on_start_processing,
                          font=("Arial", 14, "bold"), state='disabled',
                          bg="#cccccc", fg="#666666", padx=20, pady=8)
    start_button.pack(pady=15)

    # 進捗・警告表示
    if status_label:
        status_label.destroy()
    status_var = StringVar(value="")
    status_label = Label(root, textvariable=status_var, font=("Arial", 12), bg="#f0f0f0", fg="#333333")
    status_label.pack(pady=5)

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
root.geometry("1000x600")
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