# IMP302m – Đếm và phân loại kích thước vật thể bằng xử lý ảnh truyền thống

Đồ án môn **IMP302m – Image and Video Processing**, Đại học FPT.

Hệ thống tách các vật thể rời rạc (đồng xu) khỏi nền tĩnh, đếm số lượng, đo hình học và phân loại **Large / Small**.
Pipeline **không cần huấn luyện**, chỉ dùng xử lý ảnh cổ điển. Các thuật toán lõi được **tự cài đặt bằng NumPy** và đã kiểm chứng khớp với OpenCV.

**Nhóm thực hiện**

| Thành viên | Email |
|---|---|
| Ngô Quang Huy | ngoquanghuy2k5@gmail.com |
| Nguyễn Đức Mạnh | Manhndse180326@fpt.edu.vn |
| Hoàng Văn Anh | anhhvse180142@fpt.edu.vn |

Báo cáo IEEE (tiếng Anh): [`report/main.tex`](report/main.tex). Xem cách biên dịch trong [`report/README.md`](report/README.md).

---

## Kết quả chính

### 4 kịch bản ảnh giả lập (có mặt nạ chuẩn đến từng pixel)

| Kịch bản | Thách thức | Đếm | Large / Small | IoU mặt nạ |
|---|---|---|---|---|
| Case 1 – Ideal | Nền sáng đều, vật tách rời | 6 / 6 | 3 / 3 ✓ | 0.972 |
| Case 2 – Low contrast | Mức xám bị nén vào [30, 100] | 6 / 6 | 3 / 3 ✓ | 0.971 |
| Case 3 – Noise & glare | Nhiễu muối tiêu, vết lóa | 6 / 6 | 3 / 3 ✓ | 0.961 |
| Case 4 – Overlapping | Hai vật chồng lên nhau | 6 / 6 | 3 / 3 ✓ | 0.972 |

### 4 kịch bản ảnh thật (480 ảnh test, 3.386 đồng xu)

Ảnh lấy từ bộ Roboflow 100 *coins*. Mỗi ảnh được xếp vào một kịch bản chỉ dựa trên nhãn chuẩn và thống kê của ảnh (`tools/real_cases.py`), không dùng kết quả thuật toán.

| Kịch bản | Số ảnh | Gốc (A): đếm đúng / F1 | **Chốt (V): đếm đúng / F1** |
|---|---|---|---|
| R1 – Ideal | 232 | 12.9% / 0.676 | **84.1% / 0.977** |
| R2 – Low contrast | 88 | 5.7% / 0.181 | **52.3% / 0.906** |
| R3 – Textured background | 132 | 19.7% / 0.226 | **50.8% / 0.918** |
| R4 – Touching coins | 28 | 3.6% / 0.351 | **21.4% / 0.805** |
| **Tổng** | **480** | 12.9% / 0.450 | **65.4% / 0.940** |

Trên toàn tập test, cấu hình chốt V đạt:
- đếm lệch tối đa ±1 ở 84.2% số ảnh;
- sai số đếm trung bình (MAE) 0.77 xu/ảnh;
- Precision 0.921, Recall 0.959.

Tốc độ khoảng **65 ms/ảnh** 640×640 với OpenCV, khoảng 197 ms/ảnh với bản NumPy tự cài.
Mọi tham số được chọn trên tập **dev**, tập **test** chỉ chạy một lần.

---

## Pipeline

```
Ảnh màu ─► GĐ1 Tiền xử lý ─► GĐ2 Phân đoạn ─► GĐ3 Gán nhãn & đặc trưng ─► GĐ4 Phân loại
           resize, Gray,      Otsu / K-Means,    2 lượt quét + Union-Find,    Otsu 1-D trên diện tích
           FSHS, Median,      Opening, Closing,  lọc A_min, Watershed,        → Large / Small
           Gaussian           lấp lỗ             diện tích, chu vi, ...
```

**Nhánh ảnh thật** (`--preset real`) thay GĐ2 bằng phương pháp phát hiện thay đổi:
1. Ước lượng nền B bằng lọc trung vị cửa sổ lớn.
2. Tính ảnh khác biệt D = ‖I − B‖ trong không gian màu Lab, rồi phân ngưỡng bằng Triangle.
3. Bỏ các vùng chạm biên ảnh.
4. Tách vật dính bằng Watershed, chỉ áp dụng cho vùng đủ tròn.
5. Lọc theo hình dạng và diện tích.

**Kiểm chứng phần tự cài đặt** (`tools/verify_impl.py`, trên ảnh Case 3):

| Thuật toán | Kết quả so với OpenCV |
|---|---|
| Lọc trung vị 5×5 | khác biệt tối đa = 0 |
| Lọc Gaussian 5×5 | khác biệt tối đa = 1 mức xám |
| Ngưỡng Otsu | cùng T* = 186 |
| Erosion, Dilation | trùng khớp hoàn toàn |
| Gán nhãn 4 và 8 liên thông | cùng 35 vùng, cùng phân hoạch |

---

## Cài đặt

Cần Python ≥ 3.10.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Các lệnh bên dưới viết cho Windows. Trên Linux/macOS, thay `.\.venv\Scripts\python.exe` bằng `python` và đổi `\` thành `/`.

## Chạy nhanh

```powershell
# Ảnh giả lập 4 kịch bản (đã có sẵn trong repo; lệnh này để sinh lại)
.\.venv\Scripts\python.exe tools\generate_synthetic.py

# Chạy pipeline trên cả thư mục: tự đọc ground_truth.json và in bảng đánh giá
.\.venv\Scripts\python.exe main.py data\synthetic --out outputs\synthetic

# Ảnh thật: cấu hình chốt V trên tập test
.\.venv\Scripts\python.exe main.py data\real\hf_coins --preset real --subset test --save minimal --backend opencv --out outputs\real_test

# Một ảnh bất kỳ
.\.venv\Scripts\python.exe main.py duong_dan\anh.jpg --out outputs\anh_cua_toi
```

Tùy chọn thường dùng (xem đầy đủ: `python main.py -h`):

| Tùy chọn | Ý nghĩa |
|---|---|
| `--method otsu\|kmeans\|bgsub` | Thuật toán phân đoạn |
| `--preset real` | Cấu hình chốt V cho ảnh thật |
| `--classify otsu\|mean\|gap` | Cách chọn ngưỡng diện tích Large/Small |
| `--area-threshold N` | Dùng ngưỡng diện tích cố định (px) |
| `--backend numpy\|opencv` | Dùng bản tự cài hay OpenCV (kết quả như nhau, OpenCV nhanh hơn) |
| `--no-split`, `--no-fill`, `--flatten` | Tắt Watershed / tắt lấp lỗ / bật san phẳng ánh sáng |
| `--save all\|minimal\|none` | Lượng hình kết quả được lưu |
| `--subset dev\|test`, `--limit N` | Chạy tập con |

## Tái lập toàn bộ thí nghiệm và hình trong báo cáo

```powershell
# 1. Tải ảnh thật (~56 MB, không có trong repo) rồi lọc và chia dev/test.
#    Repo đã có sẵn data\real\hf_coins, chỉ cần bước này nếu muốn tạo lại từ đầu.
New-Item -ItemType Directory -Force data\downloads\hf_coins
curl.exe -L -o data\downloads\hf_coins\test.parquet https://huggingface.co/datasets/Francesco/coins-1apki/resolve/main/data/test-00000-of-00001-83ed470261ddf492.parquet
.\.venv\Scripts\python.exe tools\import_hf_coins.py

# 2. Xếp ảnh thật vào các kịch bản R1–R4
.\.venv\Scripts\python.exe tools\real_cases.py

# 3. So sánh các cấu hình: chọn trên dev, kiểm tra trên test
.\.venv\Scripts\python.exe tools\experiment_real.py data\real\hf_coins dev
.\.venv\Scripts\python.exe tools\experiment_real.py data\real\hf_coins test A,C,G,N,V,X

# 4. Kiểm chứng thuật toán tự cài so với OpenCV
.\.venv\Scripts\python.exe tools\verify_impl.py

# 5. Sinh lại toàn bộ hình, biểu đồ và số liệu của báo cáo (~5 phút)
.\.venv\Scripts\python.exe tools\make_report_figures.py

# 6. Đóng gói báo cáo để tải lên Overleaf
.\.venv\Scripts\python.exe tools\make_overleaf_zip.py
```

---

## Cấu trúc thư mục

```
ipm/                    Thư viện xử lý ảnh
  config.py             Toàn bộ tham số (dataclass Config)
  preprocess.py         GĐ1: resize NN, RGB→Gray, histogram, AOD, FSHS, Median, Gaussian, trừ nền
  segmentation.py       GĐ2: Otsu, K-Means k=2, Triangle, MAD, tự nhận nền sáng/tối
  morphology.py         GĐ2: Erosion, Dilation, Opening, Closing, lấp lỗ
  labeling.py           GĐ3: gán nhãn 2 lượt quét + Union-Find, lọc A_min, bỏ vùng chạm biên
  split.py              GĐ3: Distance Transform + Watershed tách vật dính
  features.py           GĐ3: diện tích, trọng tâm, chu vi, đường kính, hướng, độ tròn, độ đặc
  classify.py           GĐ4: ngưỡng diện tích Large/Small
  pipeline.py           Điều phối các giai đoạn, đo thời gian
  evaluate.py           So với ground truth: đếm, P/R/F1, IoU
  visualize.py          Xuất hình kết quả
main.py                 Giao diện dòng lệnh
tools/
  generate_synthetic.py Sinh 4 kịch bản giả lập kèm mặt nạ chuẩn
  import_hf_coins.py    Lọc bộ ảnh thật, chia dev/test
  real_cases.py         Xếp ảnh thật vào R1–R4
  experiment_real.py    Bảng so sánh các cấu hình (ablation)
  verify_impl.py        So sánh bản tự cài với OpenCV
  make_report_figures.py Sinh hình và biểu đồ cho báo cáo
  make_overleaf_zip.py  Đóng gói main.tex + figures/ để tải lên Overleaf
  fetch_public.py       Lưu ảnh skimage.data.coins()
data/
  synthetic/            4 ảnh giả lập + ground_truth.json + gt/ (mặt nạ)
  real/hf_coins/        960 ảnh thật + ground_truth.json (khung bao) + split.json (dev/test)
  public/               Ảnh skimage coins
outputs/                Kết quả chạy mẫu (xem bên dưới)
report/
  main.tex              Báo cáo IEEE
  figures/              12 hình và biểu đồ
  data/                 Số liệu gốc của mọi bảng và biểu đồ
```

### Kết quả đầu ra cho mỗi ảnh (`outputs/<thư mục>/<tên ảnh>/`)

| File | Nội dung |
|---|---|
| `00_pipeline_overview.png` | Các bước của pipeline trên một hình |
| `01_histogram_fshs.png`, `01b_fshs_vs_threshold.png` | Histogram trước/sau FSHS; ngưỡng cố định so với Otsu |
| `02_mask_raw.png`, `02_mask_clean.png`, `02b_morphology_steps.png` | Mặt nạ nhị phân và từng bước hình thái học |
| `02c_otsu_vs_kmeans.png` | So sánh Otsu và K-Means |
| `03_label_map.png` | Bản đồ nhãn |
| `04_detection.png` | Khung bao, trọng tâm, số thứ tự, lớp Large/Small |
| `05_features.csv` | Bảng đặc trưng của từng vật thể |
| `summary.json` | Tham số, thời gian, kết quả |

Mỗi thư mục cũng có `summary.csv` tổng hợp nhiều ảnh. Các thư mục có sẵn trong repo:
- `outputs/synthetic/`: kết quả 4 kịch bản giả lập.
- `outputs/case4_nosplit/`: Case 4 khi tắt Watershed.
- `outputs/real_showcase/`: 8 ảnh thật mẫu.
- `outputs/experiment_*.txt|json`: bảng so sánh cấu hình trên dev và test.

---

## Dữ liệu

**Ảnh giả lập** (`data/synthetic/`): sinh bằng `tools/generate_synthetic.py`, có ground truth tuyệt đối. Mặt nạ chuẩn đến từng pixel nên tính được IoU.

**Ảnh thật** (`data/real/hf_coins/`): bộ [coins-1apki](https://universe.roboflow.com/object-detection/coins-1apki) của Roboflow 100, [bản Hugging Face](https://huggingface.co/datasets/Francesco/coins-1apki), giấy phép **CC BY 4.0**.
- Lấy từ split *test* của bộ gốc, giữ lại 960 ảnh chỉ chứa đồng xu, mỗi ảnh 3–30 xu.
- Chia ngẫu nhiên dev 480 / test 480 với seed 2026.
- Nhãn chỉ gồm số lượng và khung bao, **không có nhãn lớn/nhỏ**. Vì vậy trên ảnh thật chỉ đánh giá được đếm và định vị.

**Ảnh tự chụp** (khuyến nghị, để đánh giá phân loại Large/Small trên ảnh thật):
- Đặt ảnh vào `data/raw/`, kèm file `data/raw/ground_truth.json` theo mẫu `{"anh_01.jpg": {"count": 6, "large": 3, "small": 3}}`.
- Đặt điện thoại song song mặt bàn, cách 30–40 cm, tắt flash.
- Dùng nền giấy trắng hoặc vải đen.
- Chọn 2 nhóm vật có diện tích chênh nhau ≥ 1.3 lần, ví dụ xu 5000đ và 500đ.
- Chụp đủ 4 kịch bản:

| Kịch bản | Cách tạo |
|---|---|
| Case 1 – Ideal | Ánh sáng đều, vật cách nhau ≥ 1 cm |
| Case 2 – Low contrast | Tắt đèn trần hoặc giảm phơi sáng (EV −2) |
| Case 3 – Noise & glare | Rắc vụn giấy lên nền, rọi đèn pin xiên để tạo lóa |
| Case 4 – Overlapping | Đặt 2–3 cặp vật sát hoặc gối lên nhau |

---

## Giới hạn đã biết

- Đồng xu dính thành cụm từ 3 xu trở lên, hoặc dính vào bóng đổ, thường không tách được. Đây là lý do kịch bản R4 chỉ đạt 21.4% đếm đúng.
- Nền có vân mạnh (gỗ, thảm) gây ra một phần dương tính giả.
- Phân loại Large/Small dựa trên diện tích pixel nên cần ảnh chụp cùng độ cao và cùng góc.

## Trích dẫn dữ liệu

> Roboflow 100: *coins-1apki* dataset, Roboflow Universe, CC BY 4.0. https://universe.roboflow.com/object-detection/coins-1apki
