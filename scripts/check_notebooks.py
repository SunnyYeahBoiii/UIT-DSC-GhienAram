"""Kiểm mọi cell code trong notebook compile được.

Bug vừa gặp: split("\n") bỏ mất "\n" cuối dòng -> nbformat nối liền mọi dòng thành một.
JSON vẫn hợp lệ nên không có gì báo, cho tới khi Kaggle ném SyntaxError sau 12 phút
dựng dữ liệu. Kiểm ở máy dev tốn 0,1 giây.
"""
import ast, json, re, sys
from pathlib import Path

loi = 0
for f in sys.argv[1:]:
    nb = json.loads(Path(f).read_text(encoding="utf-8"))
    ncode = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        ncode += 1
        src = "".join(c["source"])
        # Gộp dòng nối bằng "\" trước, rồi thay magic bằng pass GIỮ NGUYÊN thụt lề —
        # thay mất thụt lề thì chính validator sinh SyntaxError giả.
        s = re.sub(r"\\\n\s*", " ", src)
        s = "\n".join(
            re.match(r"\s*", l).group(0) + "pass" if l.lstrip().startswith(("!", "%")) else l
            for l in s.split("\n")
        )
        try:
            ast.parse(s)
        except SyntaxError as e:
            loi += 1
            print(f"  {f} cell {i}: {e}")
            continue
        if len(c["source"]) == 1 and "\n" not in src and len(src) > 120:
            loi += 1
            print(f"  {f} cell {i}: DÍNH DÒNG ({len(src)} ký tự, 0 newline)")
    print(f"{f}: {ncode} cell code {'LỖI' if loi else 'OK'}")
sys.exit(1 if loi else 0)
