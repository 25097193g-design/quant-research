#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建医药板块研究报告 PDF。
数据流: REPORT.md (单一真相源) -> 轻量 markdown->HTML 转换 -> headless Chrome -> PDF
产出: pharma_report.pdf (求职展示版)

用法:
    cd projects/pharma_factor_workflow
    python3 build_report_pdf.py
"""
import base64
import pathlib
import re
import subprocess
import sys

BASE = pathlib.Path(__file__).resolve().parent
REPORT = BASE / "REPORT.md"
OUT_HTML = BASE / "outputs" / "pharma_report.html"
OUT_PDF = BASE / "pharma_report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

QC_FIG = BASE / "outputs" / "panel_qc_count.png"
SCATTER = BASE / "outputs" / "subperiod_ic_scatter.png"
CONCENTRATION = BASE / "outputs" / "concentration_chart.png"
EQUITY = BASE / "outputs" / "final_equity_curve.png"

# ---------------------------------------------------------------------------
# 1. 轻量 markdown -> HTML（只覆盖 REPORT.md 用到的结构）
# ---------------------------------------------------------------------------

def inline(md: str) -> str:
    """行内格式化: **bold** / `code` / [text](url) / [[ref]]"""
    md = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    md = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", md)
    md = re.sub(r"`(.+?)`", r"<code>\1</code>", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r'<span class="link">\1</span>', md)
    md = re.sub(r"\[\[([^\]]+)\]\]", r'<em class="memref">\1</em>', md)
    return md


def parse_blocks(md_text: str):
    lines = md_text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        # fenced code block
        if s.startswith("```"):
            buf, j = [], i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            blocks.append(("code", "\n".join(buf)))
            i = j + 1
            continue
        # heading
        if s.startswith("#"):
            m = re.match(r"^(#+)\s+(.*)", s)
            blocks.append(("h", len(m.group(1)), m.group(2)))
            i += 1
            continue
        # blockquote
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            blocks.append(("quote", buf))
            continue
        # table
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(("table", rows))
            continue
        # horizontal rule
        if s == "---":
            blocks.append(("hr", ""))
            i += 1
            continue
        # list (ul / ol, 支持缩进嵌套)
        if re.match(r"^\s*- ", s) or re.match(r"^\s*\d+\.\s", s):
            items = []
            while i < len(lines):
                t = lines[i].rstrip()
                ts = t.strip()
                if re.match(r"^- ", ts):
                    indent = len(t) - len(t.lstrip())
                    items.append((indent, "ul", ts[2:].strip()))
                elif re.match(r"^\d+\.\s", ts):
                    indent = len(t) - len(t.lstrip())
                    items.append((indent, "ol", re.sub(r"^\d+\.\s", "", ts)))
                else:
                    break
                i += 1
            blocks.append(("list", items))
            continue
        # paragraph (collect until next block boundary)
        buf = [s]
        i += 1
        while i < len(lines):
            t = lines[i].strip()
            if not t or t.startswith("#") or t.startswith(">") or t.startswith("|") \
               or t.startswith("```") or t == "---" or re.match(r"^\s*- ", t) \
               or re.match(r"^\s*\d+\.\s", t):
                break
            buf.append(t)
            i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


def render_list(items) -> str:
    """嵌套列表 -> HTML。同缩进同类型 = 同一列表(连续编号); 更深缩进 = 嵌套列表"""
    out, stack = [], []
    for indent, kind, text in items:
        # 关掉更深层的列表(严格大于才关, 保证同缩进列表保持连续编号 1/2/3)
        while stack and stack[-1][0] > indent:
            out.append(f"</{stack.pop()[1]}>")
        # 同缩进但类型不同(ul<->ol) -> 关掉重开
        if stack and stack[-1][0] == indent and stack[-1][1] != kind:
            out.append(f"</{stack.pop()[1]}>")
        # 当前位置没有打开的列表 -> 打开
        if not stack or stack[-1][0] != indent or stack[-1][1] != kind:
            out.append(f"<{kind}>")
            stack.append((indent, kind))
        out.append(f"<li>{inline(text)}</li>")
    while stack:
        out.append(f"</{stack.pop()[1]}>")
    return "".join(out)


def render_table(rows) -> str:
    cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    h = "".join(f"<th>{inline(c)}</th>" for c in header)
    trs = "".join(
        "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
        for row in body
    )
    return f'<div class="tblwrap"><table><thead><tr>{h}</tr></thead><tbody>{trs}</tbody></table></div>'


def render_blocks(blocks) -> str:
    html = []
    tldr_pending = False  # TL;DR 标题后的第一个 list 包成高亮框
    for b in blocks:
        kind = b[0]
        if kind == "h":
            lvl, text = b[1], b[2]
            cls = ""
            if lvl == 2:
                cls = ' class="sec"'
            elif lvl == 3 and text.startswith("发现"):
                cls = ' class="finding"'
            elif lvl == 3:
                cls = ' class="sub"'
            tag = f"h{lvl}"
            html.append(f"<{tag}{cls}>{inline(text)}</{tag}>")
            if lvl == 2 and "TL;DR" in text:
                tldr_pending = True
            continue
        if kind == "quote":
            paras = "".join(f"<p>{inline(l)}</p>" for l in b[1])
            html.append(f"<blockquote>{paras}</blockquote>")
            continue
        if kind == "table":
            html.append(render_table(b[1]))
            continue
        if kind == "list":
            if tldr_pending:
                html.append(f'<div class="tldr">{render_list(b[1])}</div>')
                tldr_pending = False
            else:
                html.append(render_list(b[1]))
            continue
        if kind == "hr":
            html.append("<hr>")
            continue
        if kind == "fig":
            html.append(b[1])
            continue
        if kind == "code":
            html.append(
                "<pre class=\"lineage\"><code>" + b[1].replace("&", "&amp;")
                .replace("<", "&lt;").replace(">", "&gt;") + "</code></pre>"
            )
            continue
        if kind == "p":
            html.append(f"<p>{inline(b[1])}</p>")
    return "".join(html)


def img_figure(png_path, caption) -> str:
    b64 = base64.b64encode(pathlib.Path(png_path).read_bytes()).decode()
    return (
        f'<figure class="fig">'
        f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


# ---------------------------------------------------------------------------
# 2. HTML 组装
# ---------------------------------------------------------------------------

CSS = """
:root{--ink:#1f2430;--muted:#5b6472;--accent:#1f5aa5;--accent-soft:#eaf1fb;
      --line:#d5dbe4;--th:#f3f6fa;--warn:#b3261e;}
@page{size:A4;margin:15mm 15mm 17mm 15mm;}
*{box-sizing:border-box;}
html{-webkit-print-color-adjust:exact;print-color-adjust:exact;}
body{font-family:"PingFang SC","Hiragino Sans GB","Arial Unicode MS","Microsoft YaHei",sans-serif;
     font-size:10.5pt;line-height:1.72;color:var(--ink);margin:0;}
.cover{border-bottom:3px solid var(--accent);padding-bottom:10px;margin-bottom:14px;}
.cover h1{font-size:20pt;margin:0 0 6px;letter-spacing:.5px;}
.cover .meta{color:var(--muted);font-size:9.5pt;margin:0;}
.cover .meta b{color:var(--ink);}
h2.sec{font-size:13.5pt;margin:20px 0 8px;padding-bottom:4px;border-bottom:2px solid var(--accent);
       color:var(--accent);}
h3.sub{font-size:11.5pt;margin:15px 0 6px;padding-left:8px;border-left:3px solid var(--accent);}
h3.finding{font-size:11.5pt;margin:16px 0 6px;padding:4px 8px;background:var(--accent-soft);
           border-left:3px solid var(--accent);border-radius:2px;}
h4{font-size:10.5pt;margin:12px 0 4px;}
p{margin:6px 0;}
ul,ol{margin:6px 0;padding-left:22px;}
li{margin:3px 0;}
strong{color:#111;}
code{font-family:Menlo,Monaco,Consolas,monospace;font-size:8.8pt;
     background:#f2f4f7;padding:1px 4px;border-radius:3px;}
pre.lineage{font-family:Menlo,Monaco,Consolas,monospace;font-size:8.6pt;line-height:1.55;
     background:#f6f8fa;border:1px solid var(--line);border-radius:4px;
     padding:10px 12px;white-space:pre;overflow-x:auto;}
.tblwrap{overflow-x:auto;margin:10px 0;page-break-inside:avoid;}
table{border-collapse:collapse;width:100%;font-size:9.3pt;}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top;word-break:break-word;overflow-wrap:anywhere;}
th{background:var(--th);font-weight:600;}
tbody tr:nth-child(even){background:#fafbfc;}
.tldr{background:var(--accent-soft);border:1px solid #c6d8ef;border-left:4px solid var(--accent);
      border-radius:4px;padding:8px 16px 8px 12px;margin:10px 0;}
.tldr li{margin:5px 0;}
blockquote{color:#333;border-left:3px solid var(--line);padding:2px 0 2px 12px;
           margin:8px 0;font-size:9.6pt;color:var(--muted);}
blockquote p{margin:2px 0;}
hr{border:none;border-top:1px solid var(--line);margin:14px 0;}
.fig{margin:12px 0;text-align:center;page-break-inside:avoid;}
.fig img{max-width:100%;border:1px solid var(--line);border-radius:4px;}
.fig figcaption{font-size:9pt;color:var(--muted);margin-top:5px;}
.link{color:var(--accent);}
.memref{color:var(--muted);font-size:9pt;}
.pagefoot{position:fixed;bottom:0;left:0;right:0;text-align:center;font-size:8pt;
     color:var(--muted);padding:4px 0;border-top:1px solid var(--line);background:#fff;}
h1,h2,h3,h4{page-break-after:avoid;}
"""


def build_html(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>A股医药生物板块因子策略研究报告（对比银行版）</title>
<style>{CSS}</style></head>
<body>
{body}
<div class="pagefoot">Li Changhao · A股医药生物板块因子策略研究报告（对比银行版）· 2026-08</div>
</body></html>"""


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def next_heading(blocks, idx):
    for j in range(idx + 1, len(blocks)):
        if blocks[j][0] == "h":
            return j
    return len(blocks)


def inject_figures(blocks, en=False):
    """四张图：QC曲线(§1末) / 分时段散点(§2.2末) / 集中度(§3末) / Top20净值(§4.1末)。
    从后往前插（位置降序），避免 insert 改变前面 index。en=True 时图注用英文。"""
    out = list(blocks)
    def hpos(level, key):
        i = next(k for k, b in enumerate(blocks) if b[0] == "h" and b[1] == level and key in b[2])
        return i, next_heading(blocks, i)
    _, j41 = hpos(3, "4.1")
    _, j3 = hpos(2, "3.")
    _, j22 = hpos(3, "2.2")
    _, j1 = hpos(2, "1.")
    if en:
        cap = ["Fig 4  Strategy NAV (in-house monthly, 2020-01~2026-06, 0.15%/period cost; vs Q5 / pharma EW / all-A EW / CSI 300)",
               "Fig 3  Concentration inverted-U: Top20 is the sweet spot (in-house vs rqalpha T+1)",
               "Fig 2  137-factor sub-period IC scatter (green = 7 iron survivors; grey = rest; ±0.10 adaptive view)",
               "Fig 1  Panel QC: monthly pharma whitelist count (184 in 2017 → stable 210-219 from 2019; no low-N problem)"]
    else:
        cap = ["图 4  策略净值曲线（自建月频 2020-01~2026-06，含 0.15%/期成本；对照 Q5/医药等权/全A/沪深300）",
               "图 3  持仓集中度倒 U 型：Top20 是甜点（自建 vs rqalpha T+1 双口径）",
               "图 2  137 因子分时段 IC 散点（绿 = 铁存活 7；灰 = 其余；视野 ±0.10 自适应）",
               "图 1  面板 QC：医药白名单逐月截面数量（2017 起 184 只 → 2019 起稳定 210-219，无低截面问题）"]
    out.insert(j41, ("fig", img_figure(EQUITY, cap[0])))
    out.insert(j3, ("fig", img_figure(CONCENTRATION, cap[1])))
    out.insert(j22, ("fig", img_figure(SCATTER, cap[2])))
    out.insert(j1, ("fig", img_figure(QC_FIG, cap[3])))
    return out


def main():
    global REPORT, OUT_HTML, OUT_PDF
    if len(sys.argv) > 1:  # 用法: python3 build_report_pdf.py [REPORT_EN.md] → pharma_report_en.pdf
        REPORT = BASE / sys.argv[1]
        stem = sys.argv[1].rsplit(".", 1)[0].lower().replace("report", "pharma_report")
        OUT_PDF = BASE / f"{stem}.pdf"
        OUT_HTML = BASE / "outputs" / f"{stem}.html"
    if not REPORT.exists():
        sys.exit(f"找不到 {REPORT}")
    blocks = parse_blocks(REPORT.read_text(encoding="utf-8"))
    blocks = inject_figures(blocks, en=("EN" in REPORT.name.upper()))
    body = render_blocks(blocks)
    html = build_html(body)
    OUT_HTML.write_text(html, encoding="utf-8")

    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--virtual-time-budget=8000",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    size = OUT_PDF.stat().st_size
    print(f"OK  {OUT_PDF}  ({size/1024:.0f} KB)  <- {REPORT.name}")
    print(f"    HTML 中间产物: {OUT_HTML}")


if __name__ == "__main__":
    main()
