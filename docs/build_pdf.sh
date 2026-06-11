#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/build docs/figures

# Rasterize dense-error SVG figures with a real browser renderer.
# macOS Quick Look thumbnails can crop wide SVGs into square previews, so avoid it.
CHROME=""
for candidate in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium" \
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"; do
  if [[ -x "$candidate" ]]; then
    CHROME="$candidate"
    break
  fi
done

for f in \
  out/opm600/j1800-expansion-final-plots/angular/*.svg \
  out/opm600/j1800-expansion-final-plots/native/*.svg \
  out/opm600/j1800-expansion-final-plots/native-opm-vs-swiss/*.svg; do
  base="$(basename "$f" .svg)"
  out="docs/figures/${base}.png"
  if [[ ! -f "$out" || "$f" -nt "$out" ]]; then
    if [[ -n "$CHROME" ]]; then
      read -r width height < <(python3 - "$PWD/$f" "$PWD/docs/build/${base}.html" <<'PY'
from pathlib import Path
import html
import re
import sys
svg = Path(sys.argv[1])
out = Path(sys.argv[2])
text = svg.read_text(errors='ignore')[:2048]
tag = re.search(r'<svg[^>]*>', text)
width, height = 920, 460
if tag:
    wm = re.search(r'\bwidth="([0-9.]+)', tag.group(0))
    hm = re.search(r'\bheight="([0-9.]+)', tag.group(0))
    if wm and hm:
        width, height = int(float(wm.group(1))), int(float(hm.group(1)))
width *= 2
height *= 2
out.write_text(f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{{margin:0;padding:0;width:{width}px;height:{height}px;overflow:hidden;background:white;}}
img{{display:block;width:{width}px;height:{height}px;}}
</style></head><body><img src="file://{html.escape(str(svg))}"></body></html>''')
print(width, height)
PY
)
      "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
        --window-size="${width},${height}" \
        --screenshot="$PWD/$out" \
        "file://$PWD/docs/build/${base}.html" >/tmp/opm-chrome-${base}.out 2>&1
    else
      echo "error: no Chrome/Chromium found for reliable SVG rasterization" >&2
      exit 1
    fi
  fi
done

python3 - <<'PY'
from pathlib import Path
src = Path('docs/opm-short-paper-zh.md')
text = src.read_text()
lines = text.splitlines()
# The PDF template owns the title block; strip it from the Markdown body.
if lines and lines[0].startswith('# '):
    lines = lines[1:]
while lines and not lines[0].startswith('## '):
    lines = lines[1:]
text = '\n'.join(lines) + '\n'

repls = {
    '../out/opm600/j1800-expansion-final-plots/angular/mercury-dense-error.svg': 'figures/mercury-dense-error.png',
    '../out/opm600/j1800-expansion-final-plots/angular/moon-dense-error.svg': 'figures/moon-dense-error.png',
    '../out/opm600/j1800-expansion-final-plots/angular/neptune-dense-error.svg': 'figures/neptune-dense-error.png',
    '../out/opm600/j1800-expansion-final-plots/native/saturn-native-position-km.svg': 'figures/saturn-native-position-km.png',
    '../out/opm600/j1800-expansion-final-plots/native/mercury-native-velocity-mm-s.svg': 'figures/mercury-native-velocity-mm-s.png',
}
for a, b in repls.items():
    text = text.replace(a, b)
Path('docs/build/opm-short-paper-zh-pdf.md').write_text(text)
PY

pandoc docs/build/opm-short-paper-zh-pdf.md \
  --standalone \
  --from markdown+pipe_tables+tex_math_dollars \
  --to latex \
  --template docs/opm-preprint-template.tex \
  --resource-path=docs \
  -M title='OPM：一种面向有界误差部署的紧凑主要天体星历表示及其 600 年密集验证' \
  -M english-title='OPM: a compact deployable ephemeris representation for major Solar-System bodies with 600-year dense validation' \
  -M author='Rz Liu' \
  -o docs/opm-short-paper-zh.tex

python3 - <<'PY'
from pathlib import Path
import re
tex_path = Path('docs/opm-short-paper-zh.tex')
tex = tex_path.read_text()
# Inline Markdown code is used for numbers/paths in the paper; keep it visually in text font.
tex = re.sub(r'\\texttt\{([^{}]*)\}', r'{\\normalfont \1}', tex)
# The Markdown uses horizontal rules only as source separators; remove them from PDF.
tex = re.sub(r'\n?\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}\n?', '\n', tex)
# Keep images where they are introduced; Pandoc emits floating figures by default.
tex = tex.replace('\\begin{figure}\n', '\\begin{figure}[H]\n')
# Turn the abstract section into a compact shaded abstract block with consistent first-line indents.
pat = re.compile(r'\\subsection\{摘要\}\\label\{[^}]+\}\n(?P<body>.*?)(?=\\subsection\{1\. 引言\})', re.S)
m = pat.search(tex)
if m:
    body = m.group('body').strip()
    body = re.sub(r'(?m)^(\\textbf\{(?:背景|目标|方法|结果|结论|关键词)：\})', r'\\indent \1', body)
    repl = (
        '\\begin{abstractbox}\n'
        '\\noindent{\\bfseries\\color{opmblue} 摘要}\\par\\smallskip\n'
        f'{body}\n'
        '\\end{abstractbox}\n\n'
    )
    tex = tex[:m.start()] + repl + tex[m.end():]
# Convert Pandoc tokenized text code blocks back to real verbatim so indentation is preserved.
def detokenize_code_block(match):
    body = match.group('body')
    out_lines = []
    for raw in body.splitlines():
        mline = re.search(r'\\[A-Za-z]+Tok\{((?:\\.|[^{}])*)\}', raw)
        if not mline:
            continue
        line = mline.group(1)
        line = (line
            .replace(r'\_', '_')
            .replace(r'\{', '{')
            .replace(r'\}', '}')
            .replace(r'\#', '#')
            .replace(r'\%', '%')
            .replace(r'\&', '&')
            .replace(r'\textbar{}', '|')
            .replace(r'\textasciitilde{}', '~')
            .replace(r'\textasciicircum{}', '^'))
        out_lines.append(line)
    return '\\begin{Shaded}\n\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\footnotesize,xleftmargin=0.8em]\n' + '\n'.join(out_lines) + '\n\\end{Verbatim}\n\\end{Shaded}'
tex = re.sub(
    r'\\begin\{Shaded\}\n\\begin\{Highlighting\}\[\]\n(?P<body>.*?)\\end\{Highlighting\}\n\\end\{Shaded\}',
    detokenize_code_block,
    tex,
    flags=re.S,
)
# References should be hanging paragraphs, not mixed normal paragraph indents.
ref_pat = re.compile(r'(\\subsection\{参考文献\}\\label\{[^}]+\}\n)(?P<body>.*?)(?=\\end\{document\})', re.S)
ref = ref_pat.search(tex)
if ref:
    body = ref.group('body').strip()
    body = re.sub(r'(?m)^\\phantomsection\s*\n?', '', body)
    repl = (
        ref.group(1)
        + '\\begingroup\n'
        + '\\small\n'
        + '\\setlength{\\parindent}{0pt}\n'
        + '\\setlength{\\parskip}{0.35em}\n'
        + '\\hangindent=1.8em\\hangafter=1\n'
        + body.replace('\n\n', '\n\n\\hangindent=1.8em\\hangafter=1\n')
        + '\n\\endgroup\n'
    )
    tex = tex[:ref.start()] + repl + tex[ref.end():]
tex_path.write_text(tex)
PY

(
  cd docs
  tectonic opm-short-paper-zh.tex
)
