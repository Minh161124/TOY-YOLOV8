import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import time
import csv
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# --- PHẦN 1: CLASS XỬ LÝ LƯU TRỮ (LOGGING) ---
class ToyLogger:
    def __init__(self, filename='lich_su_do_choi.csv'):
        self.filename = filename
        self.initialize_csv()

    def initialize_csv(self):
        """Tạo file CSV nếu chưa tồn tại"""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Cột: Thời gian, Tổng số lượng, Chi tiết
                writer.writerow(['ThoiGian', 'TongSoLuong', 'ChiTiet'])

    def save_log(self, detections):
        """Lưu kết quả nhận diện vào CSV"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_count = sum(detections.values())
        
        details_str = "; ".join([f"{k}: {v}" for k, v in detections.items()])

        try:
            with open(self.filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([now, total_count, details_str])
            return True
        except Exception as e:
            print(f"Lỗi lưu file: {e}")
            return False

    def get_history_dataframe(self):
        """Đọc dữ liệu từ CSV lên để hiển thị"""
        if os.path.exists(self.filename):
            try:
                return pd.read_csv(self.filename)
            except:
                return None
        return None

    def export_to_excel(self, save_path):
        """Xuất báo cáo sang Excel"""
        df = self.get_history_dataframe()
        if df is not None:
            try:
                df.to_excel(save_path, index=False, sheet_name='ChiTiet')
                return True
            except Exception as e:
                print(e)
                return False
        return False

    def clear_csv(self):
        """Xóa toàn bộ dữ liệu trong file CSV (Giữ lại tiêu đề)"""
        try:
            with open(self.filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['ThoiGian', 'TongSoLuong', 'ChiTiet'])
            return True
        except Exception as e:
            print(f"Lỗi xóa file: {e}")
            return False

# --- PHẦN 2: LOAD MODEL ---
model_path = os.path.join('model', 'last.pt') 
model = None

try:
    print(f"Đang tải model từ: {model_path}")
    if not os.path.exists(model_path) and not model_path.endswith('yolov8n.pt'):
        print("Không tìm thấy model custom, đang tải yolov8n.pt mặc định...")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(model_path)
    print("✅ Đã load model thành công!")
except Exception as e:
    print("❌ LỖI LOAD MODEL:")
    print(e)

# --- PHẦN 3: GIAO DIỆN CHÍNH ---
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ Thống Giám Sát Đồ Chơi Thông Minh")
        self.window.geometry("1000x800")
        
        # Khởi tạo Logger
        self.logger = ToyLogger()
        self.last_save_time = time.time()
        self.save_interval = 3.0 

        self.current_image_path = None
        self.cap = None      
        self.is_cam_on = False 

        # --- GIAO DIỆN ---
        tk.Label(window, text="PHẦN MỀM NHẬN DIỆN & THỐNG KÊ ĐỒ CHƠI", 
                 font=("Arial", 18, "bold"), fg="#333").pack(pady=10)

        # Khung hiển thị ảnh/webcam
        self.lbl_image = tk.Label(window, text="Màn hình hiển thị", bg="#dcdcdc", width=90, height=28)
        self.lbl_image.pack(pady=5)

        # Khung chứa các nút điều khiển
        control_frame = tk.Frame(window)
        control_frame.pack(pady=10)

        # Nhóm 1: Camera & Ảnh
        tk.Label(control_frame, text="Điều Khiển:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, sticky="e")
        
        tk.Button(control_frame, text="📂 Chọn Ảnh", command=self.select_image, width=12, bg="#2196F3", fg="white").grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="🔍 Nhận Diện Ảnh", command=self.detect_image, width=15, bg="#FF9800", fg="white").grid(row=0, column=2, padx=5)
        
        self.btn_webcam = tk.Button(control_frame, text="📷 Bật Webcam", command=self.toggle_camera, width=15, bg="#4CAF50", fg="white")
        self.btn_webcam.grid(row=0, column=3, padx=5)

        # Nhóm 2: Thống Kê
        tk.Label(control_frame, text="Thống Kê:", font=("Arial", 10, "bold")).grid(row=1, column=0, padx=5, pady=10, sticky="e")
        
        # Nút lớn bao gồm chức năng xem, xuất và xóa
        tk.Button(control_frame, text="📜 Quản Lý Lịch Sử & Báo Cáo", 
                  command=self.show_history_window, 
                  width=35, bg="#673AB7", fg="white", font=("Arial", 10, "bold")
                 ).grid(row=1, column=1, padx=5, columnspan=3, sticky="ew")

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- LOGIC XỬ LÝ ẢNH ---
    def select_image(self):
        if self.is_cam_on:
            self.toggle_camera() 
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            self.current_image_path = path
            self.show_image(path)

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((720, 480)) 
        self.img_tk = ImageTk.PhotoImage(img)
        self.lbl_image.config(image=self.img_tk, width=0, height=0, text="")

    def detect_image(self):
        if self.is_cam_on:
            messagebox.showwarning("Chú ý", "Vui lòng tắt Webcam trước.")
            return
        if not self.current_image_path:
            messagebox.showwarning("Chú ý", "Bạn chưa chọn ảnh nào!")
            return

        results = model(self.current_image_path)
        self.process_and_log_results(results, is_webcam=False) # Lưu log

        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(res_rgb)
        img_pil.thumbnail((720, 480))
        self.img_tk_res = ImageTk.PhotoImage(img_pil)
        self.lbl_image.config(image=self.img_tk_res)
        
        messagebox.showinfo("Hoàn tất", f"Tìm thấy {len(results[0].boxes)} đối tượng! Đã lưu vào lịch sử.")

    # --- LOGIC WEBCAM ---
    def toggle_camera(self):
        if self.is_cam_on:
            self.is_cam_on = False
            if self.cap:
                self.cap.release()
            self.lbl_image.config(image="", text="Đã tắt Webcam", bg="#dcdcdc")
            self.btn_webcam.config(text="📷 Bật Webcam", bg="#4CAF50")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở Webcam!")
                return
            self.is_cam_on = True
            self.btn_webcam.config(text="🛑 Tắt Webcam", bg="#d32f2f")
            self.update_webcam()

    def update_webcam(self):
        if self.is_cam_on and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) 
                results = model(frame, verbose=False)
                
                # Logic lưu
                self.process_and_log_results(results, is_webcam=True)

                res_plotted = results[0].plot()
                cv2.putText(res_plotted, "REC: Auto Saving...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                img_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((720, 480))
                imgtk = ImageTk.PhotoImage(image=img_pil)

                self.lbl_image.imgtk = imgtk
                self.lbl_image.config(image=imgtk, width=0, height=0, text="")

            self.window.after(10, self.update_webcam)

    def process_and_log_results(self, results, is_webcam=False):
        detections = {}
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                detections[class_name] = detections.get(class_name, 0) + 1
        
        total_objects = sum(detections.values())

        if is_webcam:
            current_time = time.time()
            if total_objects > 0 and (current_time - self.last_save_time) > self.save_interval:
                self.logger.save_log(detections)
                self.last_save_time = current_time
        elif not is_webcam and total_objects > 0:
             self.logger.save_log(detections)

    # --- CÁC TÍNH NĂNG BÁO CÁO MỚI ---
    def show_history_window(self):
        """Hiển thị cửa sổ lịch sử + Nút Xuất Excel + Nút Xóa"""
        df = self.logger.get_history_dataframe()
        
        # Tạo cửa sổ mới (Popup)
        history_win = tk.Toplevel(self.window)
        history_win.title("Quản Lý Lịch Sử")
        history_win.geometry("750x500")
        history_win.grab_set() 

        # --- PHẦN 1: BẢNG DỮ LIỆU ---
        table_frame = tk.Frame(history_win)
        table_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Định nghĩa Treeview
        columns = ('ThoiGian', 'TongSoLuong', 'ChiTiet')
        tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        tree.heading('ThoiGian', text='Thời Gian')
        tree.heading('TongSoLuong', text='Tổng Số')
        tree.heading('ChiTiet', text='Chi Tiết')
        
        tree.column('ThoiGian', width=150, anchor="center")
        tree.column('TongSoLuong', width=80, anchor="center")
        tree.column('ChiTiet', width=450)
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Hàm nội bộ để đổ dữ liệu vào bảng
        def load_data_to_tree():
            # Xóa hết dữ liệu cũ trên bảng
            for item in tree.get_children():
                tree.delete(item)
            
            # Đọc lại dữ liệu mới từ file
            current_df = self.logger.get_history_dataframe()
            if current_df is not None and not current_df.empty:
                # Đảo ngược để hiển thị mới nhất lên đầu
                for index, row in current_df.iloc[::-1].iterrows():
                    tree.insert("", "end", values=list(row))

        # Gọi hàm load lần đầu
        load_data_to_tree()

        # --- PHẦN 2: CÁC NÚT CHỨC NĂNG (DƯỚI CÙNG) ---
        btn_frame = tk.Frame(history_win)
        btn_frame.pack(side="bottom", fill="x", padx=10, pady=15)

        # Hàm Xóa Lịch Sử
        def delete_history_action():
            confirm = messagebox.askyesno("Cảnh báo", "Bạn có chắc chắn muốn XÓA TOÀN BỘ lịch sử không?\nHành động này không thể hoàn tác.")
            if confirm:
                success = self.logger.clear_csv()
                if success:
                    load_data_to_tree() # Làm mới bảng hiển thị
                    messagebox.showinfo("Thành công", "Đã xóa sạch lịch sử!")
                else:
                    messagebox.showerror("Lỗi", "Không thể xóa file (có thể file đang mở).")

        # Nút Xuất Excel
        tk.Button(
            btn_frame, text="📥 Xuất Excel", 
            command=self.export_report, 
            bg="#009688", fg="white", font=("Arial", 10, "bold"), width=20, height=2
        ).pack(side="left", padx=20, expand=True)

        # Nút Xóa Lịch Sử
        tk.Button(
            btn_frame, text="🗑️ Xóa Lịch Sử", 
            command=delete_history_action, 
            bg="#d32f2f", fg="white", font=("Arial", 10, "bold"), width=20, height=2
        ).pack(side="right", padx=20, expand=True)

    def export_report(self):
        """Logic Xuất Excel"""
        df = self.logger.get_history_dataframe()
        if df is None or df.empty:
            messagebox.showwarning("Lỗi", "Không có dữ liệu để xuất!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            success = self.logger.export_to_excel(file_path)
            if success:
                messagebox.showinfo("Thành công", f"Đã xuất báo cáo tại:\n{file_path}")
            else:
                messagebox.showerror("Lỗi", "Không thể ghi file. Vui lòng đóng file Excel nếu đang mở.")

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()