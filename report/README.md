# Báo cáo LaTeX (IEEE, tiếng Anh)

- `main.tex`: báo cáo, lớp `IEEEtran` (conference, A4), viết hoàn toàn bằng tiếng Anh
- `figures/`: 12 hình và biểu đồ (chữ tiếng Anh), sinh tự động bằng `tools/make_report_figures.py`
- `data/`: số liệu gốc của mọi bảng và biểu đồ
  - `test_per_image.csv`: kết quả từng ảnh test (cấu hình A và V)
  - `real_cases.csv`: case R1–R4 của từng ảnh thật kèm các chỉ số dùng để chia
  - `real_case_metrics.json`, `synthetic_results.json`, `timing.json`, `verify_impl.txt`
## Biên dịch (pdfLaTeX, mặc định của Overleaf)

**Overleaf:** cần một file zip chứa `main.tex` và `figures/`. File zip này không đưa lên git, nên tạo lại như sau:
```powershell
.\.venv\Scripts\python.exe tools\make_overleaf_zip.py    # -> report\IMP302m_report_overleaf.zip
```
Không nên dùng `Compress-Archive`: lệnh này ghi đường dẫn bằng dấu `\`, Overleaf sẽ không tìm thấy hình.
Sau đó vào New Project → Upload Project → chọn file zip → Recompile.

**Trên máy** (MiKTeX hoặc TeX Live), chạy 2 lần để cập nhật số hình, bảng và tài liệu tham khảo:
```
pdflatex main.tex
pdflatex main.tex
```

## Tác giả
- Khối `\author` đã có 3 thành viên: Ngô Quang Huy, Nguyễn Đức Mạnh, Hoàng Văn Anh.
- Tên có dấu được bọc trong lệnh `\vn{...}`, lệnh này dùng bảng mã T5 (gói `vntex`, Overleaf có sẵn) nên **vẫn biên dịch bằng pdfLaTeX**.
- Muốn thêm MSSV, lớp hoặc giảng viên: thêm một dòng vào `\IEEEauthorblockA` tương ứng, hoặc thêm một khối `\and \IEEEauthorblockN{...}`.

## Tái lập số liệu
```powershell
.\.venv\Scripts\python.exe tools\generate_synthetic.py
.\.venv\Scripts\python.exe tools\import_hf_coins.py
.\.venv\Scripts\python.exe tools\real_cases.py
.\.venv\Scripts\python.exe tools\experiment_real.py data\real\hf_coins dev
.\.venv\Scripts\python.exe tools\experiment_real.py data\real\hf_coins test A,C,G,N,V,X
.\.venv\Scripts\python.exe tools\make_report_figures.py
```
