#!/usr/bin/env python
"""REPORT.md -> report.html (self-contained; figures embedded as base64). Style follows
2026_08_06_syntax_final/report/1_methodological_overview.html."""
import os, re, base64, markdown
HERE = os.path.dirname(os.path.abspath(__file__)); WD = os.path.join(HERE, "..")
md = open(os.path.join(WD, "REPORT.md")).read()
def embed(m):
    p = os.path.join(WD, m.group(2))
    if not os.path.exists(p): return m.group(0)
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{m.group(1)}"><figcaption>{m.group(1)}</figcaption></figure>'
md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
title = re.search(r"^# (.+)$", md, re.M).group(1)
css = """body{font-family:Georgia,'Times New Roman',serif;max-width:900px;margin:34px auto;padding:0 22px;line-height:1.55;color:#1a1a1a}
h1,h2,h3{font-family:-apple-system,'Segoe UI',sans-serif;color:#3b2d7e} h1{font-size:24px;border-bottom:2px solid #3b2d7e;padding-bottom:4px}
h2{font-size:16px;margin-top:26px;border-bottom:1px solid #e0dcef;padding-bottom:3px} a{color:#1e6091}
figure{margin:14px 0} img{max-width:100%;border:1px solid #e0dcef;border-radius:8px} figcaption{font-size:13px;color:#666;margin-top:5px}
code{background:#f3f1fb;padding:1px 5px;border-radius:3px;font-size:90%} table{border-collapse:collapse;font-size:13px;font-family:-apple-system,'Segoe UI',sans-serif}
th,td{border-bottom:1px solid #e0dcef;padding:4px 8px;text-align:left} th{background:#f3f1fb} .small{color:#777;font-size:13px}"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"
open(os.path.join(WD, "report.html"), "w").write(html); print("wrote report.html")
