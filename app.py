
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import time
import threading
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import requests
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Dense, InputLayer

APP_TITLE = "UEH smarttray"
CASHIER_NAME = "Nguyễn Lê Thanh Hà"

CNN_FILES = [
    "food_model.h5",
    "food_model.keras",
    "food_model_official.h5",
]

MANUAL_IMAGE_SIZE = (640, 640)

MANUAL_FOOD_BOXES = [

    (23, 58, 173, 256),      # trên trái

    (194, 60, 407, 270),     # trứng chiên

    (474, 58, 608, 230),     # thịt kho

    (23, 346, 222, 592),     # dưới trái

    (395, 335, 615, 575), 
]

UPSELL_CLASS = "thit_kho_trung"
UPSELL_EXTRA_PRICE = 6000
UPSELL_LABEL_SUFFIX_TEMPLATE = " (+{count} Trứng)"

QR_BANK_BIN = "970422"      
QR_ACCOUNT = "0907880587"
QR_TEMPLATE = "qr_only"
QR_INFO = "Thanh%20toan%20com%20canteen%20UEH"

CLASS_NAMES = [
    "ca_hu_kho",
    "canh_chua_co_ca",
    "canh_chua_khong_ca",
    "canh_rau",
    "com_trang",
    "dau_hu_sot_ca",
    "rau_xao",
    "suon_nuong",
    "thit_kho_khong_trung",
    "thit_kho_trung",
    "trung_chien",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

COLORS = {
    "background": "#F7F7F7",
    "canvas": "#FFFFFF",
    "surface_soft": "#F2F2F2",
    "hairline": "#DDDDDD",
    "ink": "#222222",
    "body": "#3F3F3F",
    "muted": "#6A6A6A",
    "primary": "#FF385C",
    "primary_active": "#E00B41",
    "success": "#008489",
    "danger": "#D7373F",
    "blue": "#287DCC",
    "warning": "#F5A623",
}


class SafeDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)


class SafeInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop("optional", None)
        batch_shape = kwargs.pop("batch_shape", None)
        if batch_shape is not None and "shape" not in kwargs:
            kwargs["shape"] = tuple(batch_shape[1:])
        super().__init__(*args, **kwargs)


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def money(value: int) -> str:
    return f"{int(value):,} VNĐ"


def load_vietnamese_font(size: int = 18):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def safe_filename_time(prefix="file", ext="jpg"):
    return f"{prefix}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


class UEHSmartTrayApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title(APP_TITLE)
        self.geometry("1440x860")
        self.minsize(1180, 740)
        self.configure(fg_color=COLORS["background"])

        self.cnn_model = None
        self.menu = {}
        self.display_to_class = {}
        self.class_display_names = []

        self.current_image_path = None
        self.current_cv_image = None
        self.annotated_cv_image = None
        self.image_tk = None
        self.qr_tk = None

        self.detected_items = []
        self.selected_index = None
        self.total_price = 0

        self.folder_images = []
        self.folder_index = 0
        self.watch_folder = None
        self.watching = False
        self.watch_seen = set()

        for d in ["cropped_foods", "captures", "results", "bills"]:
            (app_dir() / d).mkdir(exist_ok=True)

        self._load_data_and_models()
        self._build_ui()
        self._render_bill()
        self._update_qr_placeholder()
        self._set_status("Sẵn sàng")


    def _load_data_and_models(self):
        print("=" * 60)
        print("HỆ THỐNG CANTEEN UEH ĐANG KHỞI ĐỘNG")
        print("=" * 60)
        try:
            with open(app_dir() / "menu.json", "r", encoding="utf-8") as f:
                self.menu = json.load(f)
            self.class_display_names = [self.menu[c]["name"] for c in CLASS_NAMES if c in self.menu]
            self.display_to_class = {self.menu[c]["name"]: c for c in CLASS_NAMES if c in self.menu}
            print("OK Menu loaded")
        except Exception as exc:
            messagebox.showerror("Lỗi menu", f"Không đọc được menu.json:\n{exc}")
            raise

        cnn_path = None
        for model_name in CNN_FILES:
            path = app_dir() / model_name
            if path.exists():
                cnn_path = path
                break

        if cnn_path is None:
            messagebox.showerror(
                "Lỗi CNN",
                "Không tìm thấy model CNN. Hãy đặt food_model.h5 cùng thư mục với app."
            )
            raise FileNotFoundError("CNN model not found")

        try:
            with custom_object_scope({"Dense": SafeDense, "InputLayer": SafeInputLayer}):
                self.cnn_model = load_model(str(cnn_path), compile=False)
            print(f"OK CNN loaded: {cnn_path.name}")
            print("OK")
        except Exception as exc:
            messagebox.showerror("Lỗi CNN", f"Không tải được {cnn_path.name}:\n{exc}")
            raise


    def _build_ui(self):
        self.grid_columnconfigure(0, weight=5, minsize=520)
        self.grid_columnconfigure(1, weight=3, minsize=320)
        self.grid_columnconfigure(2, weight=3, minsize=500)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_panel()
        self._build_middle_panel()
        self._build_right_panel()
        self._build_status_bar()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS["canvas"], height=70, corner_radius=0)
        header.grid(row=0, column=0, columnspan=3, sticky="nsew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        logo = ctk.CTkFrame(header, fg_color=COLORS["primary"], width=44, height=44, corner_radius=22)
        logo.grid(row=0, column=0, padx=(24, 12), pady=13)
        logo.grid_propagate(False)
        ctk.CTkLabel(logo, text="UEH", text_color="white", font=ctk.CTkFont(size=12, weight="bold")).place(relx=0.5, rely=0.5, anchor="center")

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(title_box, text="UEH smarttray", font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["ink"]).pack(anchor="w")

        self.clock_var = ctk.StringVar(value="")
        ctk.CTkLabel(header, textvariable=self.clock_var, font=ctk.CTkFont(size=13), text_color=COLORS["muted"]).grid(row=0, column=2, padx=24, sticky="e")
        self._tick_clock()

    def _build_left_panel(self):
        left = ctk.CTkFrame(self, fg_color=COLORS["canvas"], corner_radius=22, border_color=COLORS["hairline"], border_width=1)
        left.grid(row=1, column=0, sticky="nsew", padx=(18, 8), pady=(18, 12))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(left, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        toolbar.grid_columnconfigure((0,1,2), weight=1)

        self._button(toolbar, "Chọn ảnh", self.choose_image, color=COLORS["primary"]).grid(row=0, column=0, padx=(0, 5))
        self._button(toolbar, "Chọn thư mục", self.choose_folder, color=COLORS["ink"]).grid(row=0, column=1, padx=4)
        self._button(toolbar, "Chụp camera", self.capture_camera, color=COLORS["success"]).grid(row=0, column=2, padx=5)
        
        self.nav_label = ctk.CTkLabel(toolbar, text="", text_color=COLORS["muted"], font=ctk.CTkFont(size=13))
        #self.nav_label.grid(row=0, column=5, sticky="e", padx=8)
        #self._button(toolbar, "<", self.prev_folder_image, color=COLORS["surface_soft"], text_color=COLORS["ink"], width=42).grid(row=0, column=3, padx=(6, 2))
        #self._button(toolbar, ">", self.next_folder_image, color=COLORS["surface_soft"], text_color=COLORS["ink"], width=42).grid(row=0, column=4, padx=(2, 6))

        image_card = ctk.CTkFrame(left, fg_color=COLORS["surface_soft"], corner_radius=18)
        image_card.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        image_card.grid_columnconfigure(0, weight=1)
        image_card.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            image_card,
            text="Chọn ảnh khay cơm để bắt đầu",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=18),
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.scan_bottom_btn = ctk.CTkButton(
            left,
            text="QUÉT AI - NHẬN DIỆN & TÍNH TIỀN",
            command=self.run_ai,
            height=58,
            corner_radius=29,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_active"],
            text_color="white",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.scan_bottom_btn.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))

        self.file_label = ctk.CTkLabel(left, text="Chưa chọn ảnh", text_color=COLORS["muted"], font=ctk.CTkFont(size=12))
        self.file_label.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 12))

    def _build_middle_panel(self):
        mid = ctk.CTkFrame(self, fg_color=COLORS["canvas"], corner_radius=22, border_color=COLORS["hairline"], border_width=1)
        mid.grid(row=1, column=1, sticky="nsew", padx=8, pady=(18, 12))
        mid.grid_columnconfigure(0, weight=1)
        mid.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(mid, text="Danh sách món ăn", font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["ink"]).grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 12))

        self.items_frame = ctk.CTkScrollableFrame(mid, fg_color=COLORS["background"], corner_radius=18)
        self.items_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        self.items_frame.grid_columnconfigure(0, weight=1)

        edit = ctk.CTkFrame(mid, fg_color=COLORS["surface_soft"], corner_radius=18)
        edit.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))
        edit.grid_columnconfigure(0, weight=1)

        self.food_combo = ctk.CTkComboBox(edit, values=self.class_display_names, width=220, fg_color="white", border_color=COLORS["hairline"], button_color=COLORS["primary"], dropdown_fg_color="white")
        self.food_combo.grid(row=0, column=0, sticky="ew", padx=(14, 8), pady=14)
        self._button(edit, "Cập nhật", self.update_selected_item, color=COLORS["primary"], width=92).grid(row=0, column=1, padx=6, pady=14)
        self._button(edit, "+ Trứng", self.add_egg_to_selected, color=COLORS["success"], width=86).grid(row=0, column=2, padx=6, pady=14)
        self._button(edit, "Xóa", self.delete_selected_item, color=COLORS["danger"], width=66).grid(row=0, column=3, padx=(6, 14), pady=14)

        total_card = ctk.CTkFrame(mid, fg_color=COLORS["primary"], corner_radius=18)
        total_card.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        total_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(total_card, text="TỔNG CỘNG", text_color="white", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, padx=16, pady=16, sticky="w")
        self.total_var = ctk.StringVar(value="0 VNĐ")
        ctk.CTkLabel(total_card, textvariable=self.total_var, text_color="white", font=ctk.CTkFont(size=22, weight="bold")).grid(row=0, column=1, padx=16, pady=16, sticky="e")

    def _build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color=COLORS["canvas"], corner_radius=22, border_color=COLORS["hairline"], border_width=1)
        right.grid(row=1, column=2, sticky="nsew", padx=(8, 18), pady=(18, 12))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="Hóa đơn thanh toán", font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["ink"]).grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 12))

        bill_card = ctk.CTkFrame(right, fg_color=COLORS["surface_soft"], corner_radius=18)
        bill_card.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        bill_card.grid_columnconfigure(0, weight=1)
        bill_card.grid_rowconfigure(0, weight=1)

        self.bill_text = ctk.CTkTextbox(
            bill_card,
            fg_color="white",
            text_color=COLORS["ink"],
            font=ctk.CTkFont(family="Consolas", size=13),
            corner_radius=14,
            border_width=1,
            border_color=COLORS["hairline"],
            wrap="none",
        )
        self.bill_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        qr_card = ctk.CTkFrame(right, fg_color="white", corner_radius=18, border_color=COLORS["hairline"], border_width=1)
        qr_card.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))
        qr_card.grid_columnconfigure(0, weight=1)
        self.qr_label = ctk.CTkLabel(qr_card, text="QR sẽ hiện khi có hóa đơn", text_color=COLORS["muted"], width=150, height=150)
        self.qr_label.grid(row=0, column=0, padx=14, pady=14)
        # Không hiển thị dòng trạng thái dài để tránh mất chữ trong giao diện
        self.qr_status = ctk.CTkLabel(qr_card, text="", height=1, text_color=COLORS["muted"], font=ctk.CTkFont(size=1))
        self.qr_status.grid(row=1, column=0, padx=0, pady=0)

        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        actions.grid_columnconfigure((0, 1), weight=1)
        self._button(actions, "In bill ", self.export_bill, color=COLORS["success"]).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._button(actions, "Làm mới", self.clear_results, color=COLORS["ink"]).grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def _build_status_bar(self):
        status = ctk.CTkFrame(self, fg_color=COLORS["canvas"], height=34, corner_radius=0)
        status.grid(row=2, column=0, columnspan=3, sticky="nsew")
        status.grid_propagate(False)
        self.status_var = ctk.StringVar(value="Sẵn sàng")
        ctk.CTkLabel(status, textvariable=self.status_var, text_color=COLORS["muted"], font=ctk.CTkFont(size=12)).pack(side="left", padx=20)

    def _button(self, parent, text, command, color, text_color="white", width=128):
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=38,
            corner_radius=20,
            fg_color=color,
            hover_color=COLORS["primary_active"] if color == COLORS["primary"] else color,
            text_color=text_color,
            font=ctk.CTkFont(size=13, weight="bold"),
        )

    def _tick_clock(self):
        self.clock_var.set(dt.datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.load_image(path)

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder_images = [
            str(Path(folder) / f) for f in sorted(os.listdir(folder))
            if Path(f).suffix.lower() in IMG_EXTS
        ]
        if not self.folder_images:
            messagebox.showinfo("Thông báo", "Thư mục không có ảnh phù hợp.")
            return
        self.folder_index = 0
        self.load_image(self.folder_images[0])
        self._update_nav_label()

    def _update_nav_label(self):
        if self.folder_images:
            self.nav_label.configure(text=f"{self.folder_index + 1}/{len(self.folder_images)}")
        else:
            self.nav_label.configure(text="")

    def prev_folder_image(self):
        if not self.folder_images:
            return
        self.folder_index = max(0, self.folder_index - 1)
        self.load_image(self.folder_images[self.folder_index])
        self._update_nav_label()

    def next_folder_image(self):
        if not self.folder_images:
            return
        self.folder_index = min(len(self.folder_images) - 1, self.folder_index + 1)
        self.load_image(self.folder_images[self.folder_index])
        self._update_nav_label()

    def capture_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera", "Không mở được camera.")
            return
        messagebox.showinfo("Camera", "Nhấn SPACE để chụp, ESC để hủy.")
        captured = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, "SPACE: Chup | ESC: Huy", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 80), 2)
            cv2.imshow("UEH SmartTray Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                captured = frame.copy()
                break
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured is not None:
            path = app_dir() / "captures" / safe_filename_time("capture", "jpg")
            cv2.imwrite(str(path), captured)
            self.load_image(str(path))
            self.after(300, self.run_ai)

    def toggle_watch_folder(self):
        if self.watching:
            self.watching = False
            self.watch_btn.configure(text="Theo dõi folder")
            self._set_status("Đã dừng theo dõi folder")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.watch_folder = folder
        self.watch_seen = set(os.listdir(folder))
        self.watching = True
        self.watch_btn.configure(text="Dừng theo dõi")
        threading.Thread(target=self._watch_folder_loop, daemon=True).start()
        self._set_status(f"Đang theo dõi: {folder}")

    def _watch_folder_loop(self):
        while self.watching and self.watch_folder:
            time.sleep(2)
            try:
                current = set(os.listdir(self.watch_folder))
                new_files = sorted(current - self.watch_seen)
                self.watch_seen = current
                for f in new_files:
                    if Path(f).suffix.lower() in IMG_EXTS:
                        path = str(Path(self.watch_folder) / f)
                        self.after(0, lambda p=path: self._auto_process_watched(p))
            except Exception:
                pass

    def _auto_process_watched(self, path):
        self.load_image(path)
        self.after(500, self.run_ai)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return
        self.current_image_path = path
        self.current_cv_image = img
        self.annotated_cv_image = None
        self.detected_items.clear()
        self.selected_index = None
        self.total_price = 0
        self.file_label.configure(text=Path(path).name)
        self._show_image(img)
        self._refresh_items_list()
        self._recalculate()
        self._render_bill()
        self._update_qr_placeholder()
        self._set_status(f"Đã chọn ảnh: {Path(path).name}")

    def _show_image(self, image_cv):
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        max_w, max_h = 760, 580
        pil.thumbnail((max_w, max_h), Image.LANCZOS)
        self.image_tk = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.image_label.configure(image=self.image_tk, text="")

    def run_ai(self):
        if self.current_cv_image is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy chọn ảnh trước.")
            return
        self.scan_bottom_btn.configure(state="disabled", text="ĐANG QUÉT...")
        self._set_status("Đang nhận diện món ăn...")
        threading.Thread(target=self._run_ai_worker, daemon=True).start()

    def _run_ai_worker(self):
        try:
    
            image_cv_original = self.current_cv_image.copy()

            fixed_w, fixed_h = MANUAL_IMAGE_SIZE
            image_cv = cv2.resize(image_cv_original, (fixed_w, fixed_h))

            items = []
            crop_dir = app_dir() / "cropped_foods"

            for old in crop_dir.glob("crop_*.jpg"):
                try:
                    old.unlink()
                except Exception:
                    pass

            print("\n" + "=" * 60)
            print("KẾT QUẢ NHẬN DIỆN CNN + TỌA ĐỘ CROP")
            print(f"File: {Path(self.current_image_path).name if self.current_image_path else 'Camera'}")
            print("-" * 60)
            print(f"{'STT':<5}{'Món ăn':<30}{'Giá':>12}{'Tin cậy':>10}")
            print("-" * 60)

            for idx, box in enumerate(MANUAL_FOOD_BOXES):
                x1, y1, x2, y2 = box

                # Giới hạn tọa độ trong ảnh
                x1 = max(0, min(x1, fixed_w - 1))
                y1 = max(0, min(y1, fixed_h - 1))
                x2 = max(0, min(x2, fixed_w))
                y2 = max(0, min(y2, fixed_h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image_cv[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                cv2.imwrite(str(crop_dir / f"crop_{idx + 1}.jpg"), crop)

                crop_resized = cv2.resize(crop, (224, 224))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

                crop_array = image.img_to_array(crop_rgb).astype("float32")
                crop_array = np.expand_dims(crop_array, axis=0)

                pred = self.cnn_model.predict(crop_array, verbose=0)
                pred_index = int(np.argmax(pred[0]))
                pred_class = CLASS_NAMES[pred_index]
                confidence = float(pred[0][pred_index]) * 100

                if pred_class not in self.menu:
                    continue

                info = self.menu[pred_class]
                base_name = info["name"]
                price = int(info["price"])
                extra_count = 0

                if pred_class == UPSELL_CLASS:
                    add = self._ask_extra_egg_sync(base_name)
                    if add:
                        extra_count = 1
                        price += UPSELL_EXTRA_PRICE

                name = self._compose_display_name(base_name, extra_count)

                item = {
                    "class": pred_class,
                    "base_name": base_name,
                    "name": name,
                    "price": price,
                    "base_price": int(info["price"]),
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                    "extra_egg_count": extra_count,
                }

                items.append(item)
                print(f"{len(items):<5}{name:<30}{money(price):>12}{confidence:>9.1f}%")

            self.detected_items = items
            self._recalculate()

            print("-" * 60)
            print(f"{'TỔNG TIỀN:':<35}{money(self.total_price):>24}")
            print("=" * 60 + "\n")

            self.annotated_cv_image = self._draw_boxes(image_cv.copy(), self.detected_items)
            self.save_result_image(silent=True)

            self.after(0, self._after_ai_done)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("Lỗi AI", str(exc)))
            self.after(0, self._after_ai_done)

    def _ask_extra_egg_sync(self, food_name):
        event = threading.Event()
        ans_holder = {"ans": False}

        def ask():
            ans_holder["ans"] = messagebox.askyesno(
                "Thêm trứng",
                f"Bạn có muốn dùng thêm trứng cho món {food_name} không?\n(+{UPSELL_EXTRA_PRICE:,} VNĐ)",
            )
            event.set()

        self.after(0, ask)
        event.wait()
        return ans_holder["ans"]

    def _after_ai_done(self):
        if self.annotated_cv_image is not None:
            self._show_image(self.annotated_cv_image)
        self._refresh_items_list()
        self._render_bill()
        self._update_qr_async()
        self.scan_bottom_btn.configure(state="normal", text="QUÉT AI - NHẬN DIỆN & TÍNH TIỀN")
        self._set_status(f"Đã nhận diện {len(self.detected_items)} món - Tổng {money(self.total_price)}")

   
    def _compose_display_name(self, base_name, extra_count):
        if extra_count and extra_count > 0:
            return base_name + UPSELL_LABEL_SUFFIX_TEMPLATE.format(count=extra_count)
        return base_name

    def _refresh_items_list(self):
        for child in self.items_frame.winfo_children():
            child.destroy()
        if not self.detected_items:
            ctk.CTkLabel(self.items_frame, text="Chưa có món nào", text_color=COLORS["muted"], font=ctk.CTkFont(size=15)).grid(row=0, column=0, sticky="ew", pady=36)
            return
        for idx, item in enumerate(self.detected_items):
            selected = idx == self.selected_index
            bg = COLORS["primary"] if selected else "white"
            fg = "white" if selected else COLORS["ink"]
            card = ctk.CTkFrame(self.items_frame, fg_color=bg, corner_radius=16, border_width=1, border_color=COLORS["hairline"])
            card.grid(row=idx, column=0, sticky="ew", padx=4, pady=6)
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(card, text=str(idx + 1), width=34, text_color=fg, font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=(12, 6), pady=14)
            ctk.CTkLabel(card, text=item["name"], text_color=fg, font=ctk.CTkFont(size=15, weight="bold"), anchor="w").grid(row=0, column=1, sticky="ew", padx=4, pady=14)
            ctk.CTkLabel(card, text=money(item["price"]), text_color=fg, font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=2, padx=14, pady=14)
            for widget in [card] + card.winfo_children():
                widget.bind("<Button-1>", lambda _e, i=idx: self._select_item(i))

    def _select_item(self, idx):
        self.selected_index = idx
        item = self.detected_items[idx]
        self.food_combo.set(item["base_name"])
        self._refresh_items_list()

    def update_selected_item(self):
        if self.selected_index is None:
            messagebox.showwarning("Chưa chọn", "Hãy chọn một món trong danh sách trước.")
            return
        display_name = self.food_combo.get()
        new_class = self.display_to_class.get(display_name)
        if not new_class:
            messagebox.showerror("Lỗi", "Món được chọn không hợp lệ.")
            return
        info = self.menu[new_class]
        base_name = info["name"]
        base_price = int(info["price"])
        extra_count = 0
        price = base_price
        if new_class == UPSELL_CLASS:
            extra = messagebox.askyesno("Thêm trứng", f"Bạn có muốn dùng thêm trứng cho món {base_name} không?\n(+{UPSELL_EXTRA_PRICE:,} VNĐ)")
            if extra:
                extra_count = 1
                price += UPSELL_EXTRA_PRICE
        self.detected_items[self.selected_index].update({
            "class": new_class,
            "base_name": base_name,
            "base_price": base_price,
            "name": self._compose_display_name(base_name, extra_count),
            "price": price,
            "confidence": 100.0,
            "extra_egg_count": extra_count,
        })
        self._sync_after_manual_change("Đã cập nhật món")

    def add_egg_to_selected(self):
        if self.selected_index is None:
            messagebox.showwarning("Chưa chọn", "Hãy chọn món thịt kho trứng trước.")
            return
        item = self.detected_items[self.selected_index]
        if item["class"] != UPSELL_CLASS:
            messagebox.showinfo("Không áp dụng", "Chỉ món Thịt kho trứng mới có tùy chọn thêm trứng.")
            return
        item["extra_egg_count"] = int(item.get("extra_egg_count", 0)) + 1
        item["price"] = int(item.get("base_price", self.menu[UPSELL_CLASS]["price"])) + item["extra_egg_count"] * UPSELL_EXTRA_PRICE
        item["name"] = self._compose_display_name(item["base_name"], item["extra_egg_count"])
        self._sync_after_manual_change(f"Đã thêm trứng lần {item['extra_egg_count']}")

    def delete_selected_item(self):
        if self.selected_index is None:
            messagebox.showwarning("Chưa chọn", "Hãy chọn một món để xóa.")
            return
        del self.detected_items[self.selected_index]
        self.selected_index = None
        self._sync_after_manual_change("Đã xóa món")

    def _sync_after_manual_change(self, status):
        self._recalculate()
        if self.current_cv_image is not None:
            self.annotated_cv_image = self._draw_boxes(self.current_cv_image.copy(), self.detected_items)
            self._show_image(self.annotated_cv_image)
            self.save_result_image(silent=True)
        self._refresh_items_list()
        self._render_bill()
        self._update_qr_async()
        self._print_console_bill(prefix="CẬP NHẬT")
        self._set_status(status)

    def _recalculate(self):
        self.total_price = sum(int(item["price"]) for item in self.detected_items)
        self.total_var.set(money(self.total_price))

    def _draw_boxes(self, image_cv, items):
        pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font = load_vietnamese_font(18)
        for item in items:
            x1, y1, x2, y2 = item["box"]
            label = item["name"]
            color = (255, 56, 92)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            try:
                text_box = draw.textbbox((0, 0), label, font=font)
                tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
            except Exception:
                tw, th = len(label) * 10, 20
            # Nếu box sát mép trên, đặt nhãn bên trong để không mất chữ
            if y1 - th - 12 >= 0:
                y0 = y1 - th - 12
                y_text = y0 + 4
                y1_label = y1
            else:
                y0 = y1 + 4
                y_text = y0 + 4
                y1_label = y0 + th + 12
            draw.rounded_rectangle([x1, y0, x1 + tw + 16, y1_label], radius=8, fill=color)
            draw.text((x1 + 8, y_text), label, fill=(255, 255, 255), font=font)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _bill_string(self):
        now = dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        filename = Path(self.current_image_path).name if self.current_image_path else "Chưa có ảnh"
        w = 42
        lines = [
            "=" * w,
            "      CANTEEN TRƯỜNG ĐẠI HỌC UEH",
            "             PHIẾU THANH TOÁN",
            "=" * w,
            f"Ngày    : {now}",
            f"Thu ngân: {CASHIER_NAME}",
            f"File    : {filename}",
            "-" * w,
            f"{'STT':<4}{'Tên món':<26}{'Giá':>10}",
            "-" * w,
        ]
        if not self.detected_items:
            lines.append("Chưa có món nào")
        else:
            for i, item in enumerate(self.detected_items, start=1):
                name = item["name"][:25]
                lines.append(f"{i:<4}{name:<26}{int(item['price']):>10,}")
        lines += [
            "-" * w,
            f"{'TỔNG CỘNG:':<30}{self.total_price:>10,} VNĐ",
            "=" * w,
            "           Cảm ơn bạn đã dùng bữa!",
            "=" * w,
        ]
        return "\n".join(lines)

    def _render_bill(self):
        self.bill_text.configure(state="normal")
        self.bill_text.delete("1.0", "end")
        self.bill_text.insert("1.0", self._bill_string())
        self.bill_text.configure(state="disabled")

    def _update_qr_placeholder(self):
        self.qr_tk = None
        self.qr_label.configure(image=None, text="QR sẽ hiện khi có hóa đơn")
        self.qr_status.configure(text="")

    def _update_qr_async(self):
        if self.total_price <= 0:
            self._update_qr_placeholder()
            return
        threading.Thread(target=self._qr_worker, daemon=True).start()

    def _qr_worker(self):
        try:
            url = f"https://img.vietqr.io/image/{QR_BANK_BIN}-{QR_ACCOUNT}-{QR_TEMPLATE}.png?amount={self.total_price}&addInfo={QR_INFO}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            pil = Image.open(io.BytesIO(response.content)).convert("RGB")
            pil = pil.resize((150, 150), Image.LANCZOS)
            self.qr_tk = ImageTk.PhotoImage(pil)
            self.after(0, lambda: self.qr_label.configure(image=self.qr_tk, text=""))
        except Exception as exc:
            print(f"[QR ERROR] {exc}")
            self.after(0, lambda: self.qr_label.configure(image=None, text="Không tải được QR"))

    
    def export_bill(self):
        if not self.detected_items:
            messagebox.showwarning("Chưa có hóa đơn", "Hãy quét AI trước khi in/lưu bill.")
            return
        default_name = f"bill_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name, filetypes=[("Text file", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._bill_string())
        messagebox.showinfo("Thành công", f"Đã lưu hóa đơn:\n{path}")
        self._set_status("Đã lưu hóa đơn")

    def save_result_image(self, silent=False):
        if self.annotated_cv_image is None:
            if not silent:
                messagebox.showwarning("Chưa có ảnh", "Chưa có ảnh kết quả để lưu.")
            return
        path = app_dir() / "results" / f"result_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), self.annotated_cv_image)
        if not silent:
            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả:\n{path}")

    def clear_results(self):
        self.detected_items.clear()
        self.selected_index = None
        self.total_price = 0
        self.annotated_cv_image = None
        if self.current_cv_image is not None:
            self._show_image(self.current_cv_image)
        else:
            self.image_label.configure(image=None, text="Chọn ảnh khay cơm để bắt đầu")
        self._refresh_items_list()
        self._recalculate()
        self._render_bill()
        self._update_qr_placeholder()
        self._set_status("Đã làm mới")

    def _print_console_bill(self, prefix="HÓA ĐƠN"):
        print("\n" + "=" * 60)
        print(prefix)
        print("-" * 60)
        for i, item in enumerate(self.detected_items, start=1):
            print(f"{i:<4}{item['name']:<32}{money(item['price']):>14}")
        print("-" * 60)
        print(f"{'TỔNG TIỀN:':<36}{money(self.total_price):>24}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    app = UEHSmartTrayApp()
    app.mainloop()
