"""Pack report/main.tex + report/figures/ into a zip file for uploading to Overleaf.

Uses zipfile (paths like figures/xxx.png) because Compress-Archive in Windows PowerShell 5.1
writes backslashes into the entry names, which Overleaf (Linux) does not treat as folders.
Run:  python tools/make_overleaf_zip.py   ->  report/IMP302m_report_overleaf.zip
"""
import sys
import zipfile
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "report"
OUT = REPORT / "IMP302m_report_overleaf.zip"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    files = [REPORT / "main.tex"] + sorted(p for p in (REPORT / "figures").iterdir() if p.is_file())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(REPORT).as_posix())
    print(f"Created {OUT} ({len(files)} files)")


if __name__ == "__main__":
    main()
