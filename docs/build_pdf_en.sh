#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/build docs/figures

# Rasterize SVG figures with a real browser renderer.  The English paper uses
# the same figure PNGs as the Chinese draft.
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
src = Path('docs/opm-short-paper-en.md')
text = src.read_text()
lines = text.splitlines()
# The PDF template owns the title block; strip it from the Markdown body.
if lines and lines[0].startswith('# '):
    lines = lines[1:]
while lines and not lines[0].startswith('## '):
    lines = lines[1:]
text = '\n'.join(lines) + '\n'
Path('docs/build/opm-short-paper-en-pdf.md').write_text(text)
PY

pandoc docs/build/opm-short-paper-en-pdf.md \
  --standalone \
  --from markdown+pipe_tables+tex_math_dollars \
  --to latex \
  --template docs/opm-two-column-preprint-template.tex \
  --resource-path=docs \
  -M title='OPM: A Compact Deployable Ephemeris Representation for Major Solar-System Bodies with 600-Year Dense Validation' \
  -M author='Rz Liu' \
  -o docs/opm-short-paper-en.tex

python3 - <<'PY'
from pathlib import Path
import re
tex_path = Path('docs/opm-short-paper-en.tex')
tex = tex_path.read_text()
# Inline Markdown code is used for numbers/paths in the paper; keep it visually in text font.
tex = re.sub(r'\\texttt\{([^{}]*)\}', r'{\\normalfont \1}', tex)
# The Markdown uses horizontal rules only as source separators; remove them from PDF.
tex = re.sub(r'\n?\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}\n?', '\n', tex)
# Keep images where they are introduced; Pandoc emits floating figures by default.
tex = tex.replace('\\begin{figure}\n', '\\begin{figure}[H]\n')
# The neutral English PDF is two-column; Pandoc longtable cannot be used in
# twocolumn mode. Convert wide numerical/image tables to full-width floats, but
# keep compact data tables at their natural single-column size.
def rewrite_longtables(source):
    begin_token = '\\begin{longtable}'
    appendix_start = source.find('\\subsection{Appendix A.')
    out = []
    pos = 0
    while True:
        start = source.find(begin_token, pos)
        if start < 0:
            out.append(source[pos:])
            break
        out.append(source[pos:start])
        cursor = start + len(begin_token)
        if cursor < len(source) and source[cursor] == '[':
            depth = 1
            cursor += 1
            while cursor < len(source) and depth:
                if source[cursor] == '[':
                    depth += 1
                elif source[cursor] == ']':
                    depth -= 1
                cursor += 1
        if cursor >= len(source) or source[cursor] != '{':
            out.append(source[start:cursor])
            pos = cursor
            continue
        spec_start = cursor + 1
        depth = 1
        cursor += 1
        while cursor < len(source) and depth:
            if source[cursor] == '{':
                depth += 1
            elif source[cursor] == '}':
                depth -= 1
            cursor += 1
        spec = source[spec_start:cursor - 1]
        end_token = '\\end{longtable}'
        end = source.find(end_token, cursor)
        if end < 0:
            out.append(source[start:])
            pos = len(source)
            break
        body = source[cursor:end]
        body = re.sub(r'(?m)^\\end(?:firsthead|head|foot|lastfoot)\s*\n?', '', body)
        col_count = max(
            spec.count('\\raggedright') + spec.count('\\raggedleft') + spec.count('\\centering'),
            len(re.findall(r'(?<![A-Za-z])[lrc](?![A-Za-z])', spec)),
        )
        in_appendix = appendix_start >= 0 and start >= appendix_start
        has_graphics = '\\includegraphics' in body
        is_appendix_thumbnail = in_appendix and has_graphics and col_count <= 2
        is_wide = (has_graphics and not is_appendix_thumbnail) or col_count > 4
        before = out[-1]
        caption_match = re.search(
            r'(?P<caption>\n\\textbf\{(?:Table|Figure) [^}]+\}[^\n]*(?:\n(?!\n|\\textbf\{|\\begin\{|\\sub|\\section).*)*)\n\n(?P<prefix>\{\\def\\LTcaptype\{none\} % do not increment counter\n)$',
            before,
        )
        caption = ''
        if caption_match:
            caption = caption_match.group('caption').strip()
            out[-1] = before[:caption_match.start('caption')] + caption_match.group('prefix')
        caption_tex = ('\\begin{center}\\small ' + caption + '\\end{center}\n') if caption else ''
        if is_wide and not in_appendix:
            out.append(
                '\\begin{table*}[t]\n\\centering\n\\scriptsize\n'
                + caption_tex +
                '\\resizebox{\\textwidth}{!}{%\n'
                f'\\begin{{tabular}}{{{spec}}}'
                + body
                + '\\end{tabular}%\n}\n\\end{table*}'
            )
        else:
            width = '\\textwidth' if in_appendix else '\\linewidth'
            wrapper_begin = f'\\resizebox{{{width}}}{{!}}{{%\n' if is_wide else ''
            wrapper_end = '\\end{tabular}%\n}\n' if is_wide else '\\end{tabular}\n'
            out.append(
                '\\begin{table}[H]\n\\centering\n\\scriptsize\n'
                + caption_tex + wrapper_begin +
                f'\\begin{{tabular}}{{{spec}}}'
                + body
                + wrapper_end + '\\end{table}'
            )
        pos = end + len(end_token)
    return ''.join(out)

tex = rewrite_longtables(tex)
# Keep the main paper two-column through References, then switch appendices to
# one-column layout so wide configuration tables and figure grids stay in order.
tex = tex.replace('\\subsection{Appendix A.', '\\FloatBarrier\n\\clearpage\n\\onecolumn\n\\subsection{Appendix A.', 1)
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
    return '\\begin{Shaded}\n\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\scriptsize,xleftmargin=0.6em]\n' + '\n'.join(out_lines) + '\n\\end{Verbatim}\n\\end{Shaded}'
tex = re.sub(
    r'\\begin\{Shaded\}\n\\begin\{Highlighting\}\[\]\n(?P<body>.*?)\\end\{Highlighting\}\n\\end\{Shaded\}',
    detokenize_code_block,
    tex,
    flags=re.S,
)
# References should be hanging paragraphs, not mixed normal paragraph indents.
ref_pat = re.compile(r'(\\subsection\{References\}\\label\{[^}]+\}\n)(?P<body>.*?)(?=\\end\{document\})', re.S)
ref = ref_pat.search(tex)
if ref:
    body = ref.group('body').strip()
    appendix_marker = '\\FloatBarrier\n\\clearpage\n\\onecolumn\n\\subsection{Appendix A.'
    appendix_tail = ''
    appendix_pos = body.find(appendix_marker)
    if appendix_pos >= 0:
        appendix_tail = body[appendix_pos:]
        body = body[:appendix_pos].strip()
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
        + appendix_tail
    )
    tex = tex[:ref.start()] + repl + tex[ref.end():]
tex_path.write_text(tex)
PY

(
  cd docs
  tectonic opm-short-paper-en.tex
)
