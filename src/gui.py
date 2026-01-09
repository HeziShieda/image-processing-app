import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sv_ttk
from PIL import Image, ImageTk

from io_utils import pil_to_np, np_to_pil
from histogram import get_hist_image
from color_space import *
from filters import *

class ImageApp:
    def __init__(self, root):
        self.root = root
        root.title('Chuyển đổi hệ màu & Nâng cao chất lượng ảnh cơ bản')
        self.orig_img = None
        self.orig_arr = None
        self.proc_arr = None

        # Frames
        self.left = ttk.Frame(root)
        self.right = ttk.Frame(root)
        self.left.pack(side='left', padx=10, pady=10)
        self.right.pack(side='right', padx=10, pady=10, fill='y')

        # ===== Ảnh gốc + Histogram =====
        frame_orig = ttk.Frame(self.left)
        frame_orig.pack(padx=5, pady=5, fill="x")

        # khung ảnh gốc
        subframe_orig = ttk.Frame(frame_orig)
        subframe_orig.pack(side="left", padx=5)
        ttk.Label(subframe_orig, text="Ảnh gốc").pack()
        self.canvas_orig = tk.Canvas(subframe_orig, width=360, height=360, bg='black')
        self.canvas_orig.pack()

        # histogram ảnh gốc
        subframe_horig = ttk.Frame(frame_orig)
        subframe_horig.pack(side="left", padx=5)
        ttk.Label(subframe_horig, text="Histogram gốc").pack()
        self.canvas_hist_orig = tk.Canvas(subframe_horig, width=360, height=360, bg='white')
        self.canvas_hist_orig.pack()

        # ===== Ảnh xử lý + Histogram =====
        frame_proc = ttk.Frame(self.left)
        frame_proc.pack(padx=5, pady=5, fill="x")

        # khung ảnh xử lý
        subframe_proc = ttk.Frame(frame_proc)
        subframe_proc.pack(side="left", padx=5)
        ttk.Label(subframe_proc, text="Ảnh đã xử lý").pack()
        self.canvas_proc = tk.Canvas(subframe_proc, width=360, height=360, bg='black')
        self.canvas_proc.pack()

        # histogram ảnh xử lý
        subframe_hproc = ttk.Frame(frame_proc)
        subframe_hproc.pack(side="left", padx=5)
        ttk.Label(subframe_hproc, text="Histogram xử lý").pack()
        self.canvas_hist_proc = tk.Canvas(subframe_hproc, width=360, height=360, bg='white')
        self.canvas_hist_proc.pack()

        # Controls
        btn_frame = ttk.Frame(self.right)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text='Tải ảnh lên', command=self.load_image).grid(row=0, column=0, sticky='we')
        ttk.Button(btn_frame, text='Lưu ảnh', command=self.save_processed).grid(row=0, column=1, sticky='we')

        # Conversion options
        conv_frame = ttk.LabelFrame(self.right, text='Chuyển đổi hệ màu')
        conv_frame.pack(fill='x', pady=5)
        self.conv_var = tk.StringVar(value='Không')
        conv_opts = ['Không', 'RGB->HSV', 'HSV->RGB', 'RGB->YCbCr', 'YCbCr->RGB','RGB->CMYK', 'CMYK->RGB', 'RGB->Gray']
        ttk.Combobox(conv_frame, values=conv_opts, textvariable=self.conv_var, state='readonly').pack(fill='x', padx=5,pady=5)
        ttk.Button(conv_frame, text='Áp dụng chuyển đổi', command=self.apply_conversion).pack(fill='x', padx=5, pady=5)

        # Filters
        filt_frame = ttk.LabelFrame(self.right, text='Lọc nhiễu + Nâng cao chất lượng ảnh')
        filt_frame.pack(fill='x', pady=5)
        self.filter_var = tk.StringVar(value='Làm mờ Gaussian')
        ttk.Combobox(filt_frame, values=['Làm mờ Gaussian', 'Lọc Trung Vị', 'Làm nét', 'Lọc tùy biến', 'Cân bằng lược đồ xám', 'Cân bằng lược đồ màu'],textvariable=self.filter_var, state='readonly').pack(fill='x', padx=5, pady=5)
        ttk.Button(filt_frame, text='Áp dụng lọc', command=self.apply_filter).pack(fill='x', padx=5, pady=5)

        # Parameters
        param_frame = ttk.LabelFrame(self.right, text='Tham số bộ lọc')
        param_frame.pack(fill='x', pady=5)
        ttk.Label(param_frame, text='Kích thước cửa sổ lọc').grid(row=0, column=0)
        self.kentry = ttk.Entry(param_frame);
        self.kentry.insert(0, '5')
        self.kentry.grid(row=0, column=1)
        ttk.Label(param_frame, text='Độ lệch chuẩn').grid(row=1, column=0)
        self.sentry = ttk.Entry(param_frame);
        self.sentry.insert(0, '1.0')
        self.sentry.grid(row=1, column=1)
        ttk.Label(param_frame, text='Mức độ làm nét').grid(row=2, column=0)
        self.aentry = ttk.Entry(param_frame);
        self.aentry.insert(0, '1.0')
        self.aentry.grid(row=2, column=1)

        # Brightness/contrast
        bc_frame = ttk.LabelFrame(self.right, text='Độ sáng và độ tương phản')
        bc_frame.pack(fill='x', pady=5)
        ttk.Label(bc_frame, text='Độ sáng (-255..255)').pack()
        self.bscale = tk.Scale(bc_frame, from_=-255, to=255, orient='horizontal')
        self.bscale.set(0);
        self.bscale.pack(fill='x')
        ttk.Label(bc_frame, text='Độ tương phản (0.1..3.0)').pack()
        self.cscale = tk.Scale(bc_frame, from_=10, to=300, orient='horizontal', command=self._dummy)
        self.cscale.set(100);
        self.cscale.pack(fill='x')
        ttk.Button(bc_frame, text='Áp dụng chỉnh sửa', command=self.apply_bc).pack(fill='x', padx=5, pady=5)

        # Quick restore
        ttk.Button(self.right, text='Khôi phục ảnh gốc', command=self.reset).pack(fill='x', pady=10)

        # Image references for Tk
        self._tk_orig = None
        self._tk_proc = None
        self._tk_hist_orig = None
        self._tk_hist_proc = None

        sv_ttk.set_theme("dark")

    def _dummy(self, v):
        pass

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files','*.png;*.jpg;*.jpeg;*.bmp;*.tiff'),('All files','*.*')])
        if not path:
            return
        img = Image.open(path).convert('RGB')
        self.orig_img = img
        self.orig_arr = pil_to_np(img)
        self.proc_arr = self.orig_arr.copy()
        self.show_images()

    def save_processed(self):
        if self.proc_arr is None:
            messagebox.showinfo('No image','No processed image to save')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png'),('JPEG','*.jpg;*.jpeg')])
        if not path:
            return
        np_to_pil(self.proc_arr).save(path)
        messagebox.showinfo('Saved', f'Saved to {path}')

    def show_images(self):
        if self.orig_arr is not None:
            img = Image.fromarray(self.orig_arr)
            img.thumbnail((360,360))
            self._tk_orig = ImageTk.PhotoImage(img)
            self.canvas_orig.delete('all')
            self.canvas_orig.create_image(180,180, image=self._tk_orig)
            # histogram gốc
            himg = get_hist_image(self.orig_arr)
            self._tk_hist_orig = ImageTk.PhotoImage(himg)
            self.canvas_hist_orig.delete('all')
            self.canvas_hist_orig.create_image(180, 100, image=self._tk_hist_orig)
        if self.proc_arr is not None:
            img = Image.fromarray(self.proc_arr)
            img.thumbnail((360,360))
            self._tk_proc = ImageTk.PhotoImage(img)
            self.canvas_proc.delete('all')
            self.canvas_proc.create_image(180,180, image=self._tk_proc)
            # histogram xử lý
            himg = get_hist_image(self.proc_arr)
            self._tk_hist_proc = ImageTk.PhotoImage(himg)
            self.canvas_hist_proc.delete('all')
            self.canvas_hist_proc.create_image(180, 100, image=self._tk_hist_proc)

    def reset(self):
        if self.orig_arr is not None:
            self.proc_arr = self.orig_arr.copy()
            self.show_images()

    def apply_conversion(self):
        if self.proc_arr is None:
            return
        op = self.conv_var.get()
        arr = self.proc_arr
        if op == 'Không':
            return
        elif op == 'RGB->HSV':
            hsv = rgb_image_to_hsv(arr)
            # show as visualized HSV (H normalized to 0..255, S,V to 0..255)
            H = (hsv[...,0] / 360.0 * 255.0).astype(np.uint8)
            S = (hsv[...,1] * 255.0).astype(np.uint8)
            V = (hsv[...,2] * 255.0).astype(np.uint8)
            self.proc_arr = np.stack([H,S,V], axis=-1)
        elif op == 'HSV->RGB':
            # interpret current image as H,S,V in channels (H scaled 0..255->0..360)
            h = (arr[...,0].astype(np.float32) / 255.0) * 360.0
            s = arr[...,1].astype(np.float32) / 255.0
            v = arr[...,2].astype(np.float32) / 255.0
            hsv = np.stack([h,s,v], axis=-1)
            self.proc_arr = hsv_image_to_rgb(hsv)
        elif op == 'RGB->YCbCr':
            self.proc_arr = rgb_to_ycbcr(arr).astype(np.uint8)
        elif op == 'YCbCr->RGB':
            self.proc_arr = ycbcr_to_rgb(arr.astype(np.float32))
        elif op == 'RGB->Gray':
            gray = rgb_to_gray(arr)
            self.proc_arr = np.stack([gray,gray,gray], axis=-1)
        elif op == 'RGB->CMYK':
            self.proc_arr = rgb_to_cmyk(arr)
        elif op == 'CMYK->RGB':
            # Nếu ảnh hiện tại là CMYK 4 kênh, convert sang RGB
            if arr.shape[-1] == 4:
                self.proc_arr = cmyk_to_rgb(arr)
            else:
                messagebox.showerror("Lỗi", "Ảnh hiện tại không phải CMYK")
                return
        self.show_images()

    def apply_filter(self):
        if self.proc_arr is None:
            return
        filt = self.filter_var.get()
        k = int(self.kentry.get()) if self.kentry.get().isdigit() else 5
        sigma = float(self.sentry.get()) if self.sentry.get() else 1.0
        amount = float(self.aentry.get()) if self.aentry.get() else 1.0
        arr = self.proc_arr
        if filt == 'Làm mờ Gaussian':
            kernel = gaussian_kernel(k, sigma)
            self.proc_arr = apply_convolution(arr, kernel)
        elif filt == 'Lọc Trung Vị':
            self.proc_arr = median_filter(arr, k)
        elif filt == 'Làm nét':
            self.proc_arr = unsharp_mask(arr, kernel_size=k, sigma=sigma, amount=amount)
        elif filt == 'Lọc tùy biến':
            # example: simple edge detection kernel
            kernel = np.array([[ -1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
            self.proc_arr = apply_convolution(arr, kernel)
        elif filt == 'Cân bằng lược đồ xám':
            gray = rgb_to_gray(arr)
            eq = histogram_equalize_gray(gray)
            self.proc_arr = np.stack([eq,eq,eq], axis=-1)
        elif filt == 'Cân bằng lược đồ màu':
            self.proc_arr = histogram_equalize_color(arr)
        self.show_images()

    def apply_bc(self):
        if self.proc_arr is None:
            return
        brightness = float(self.bscale.get())
        contrast = float(self.cscale.get()) / 100.0
        self.proc_arr = adjust_brightness_contrast(self.proc_arr, brightness=brightness, contrast=contrast)
        self.show_images()
