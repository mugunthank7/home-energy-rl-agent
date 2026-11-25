"""Build docs/Final_Report.docx from docs/REPORT_DRAFT.md (needs pandoc on the PATH).

What it does, in order:
1. Reads the draft, drops the draft-only notes and the figure appendix, and turns the H1 into the title.
2. Inserts the figures from reports/figures/ with captions after the paragraphs they illustrate.
3. Highlights every [YOU ...] placeholder in yellow so none is missed.
4. Converts with pandoc using a compact reference style (10.5 pt, 0.85 inch margins, bordered tables).

Usage: python scripts/build_report_docx.py [--out docs/Final_Report.docx]
"""
import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
AUTHOR = "Mugunthan Kesavan"

# (paragraph prefix to insert after, [(figure file, caption, width in inches)])
FIGURES = [
    ("**Exploratory analysis", [
        ("eda_daily_profile.png",
         "Figure 1. Mean demand by hour of day on weekdays and weekends (REAL data, 2007-2010).", 3.8),
    ]),
    ("The 5-seed std is 1 to 3 points", [
        ("sweep_grid_alpha_gamma_heatmap.png",
         "Figure 2. Total savings (%) over learning rate and discount factor, mean of 5 seeds (validation split).", 3.3),
        ("sweep_epsilon_convergence.png",
         ("Figure 3. Convergence under four epsilon-decay schedules: training reward (left) and greedy "
          "validation reward (right), mean ± std over 5 seeds."), 6.0),
    ]),
    ("The best tabular controller saves about", [
        ("controllers_test.png",
         ("Figure 4. Test split (2010): weekly cost, weekly peak and total savings by controller. Error bars are "
          "the std over seeds for learned controllers and the standard error over weeks for deterministic ones."), 6.4),
    ]),
    ("*Table 5. Total savings on held-out seasons", [
        ("deep_rl_curves.png",
         ("Figure 5. Greedy validation reward of PPO and DQN during training, with and without the peak input "
          "(mean ± std over 5 seeds)."), 4.2),
    ]),
    ("**How far from the best possible", [
        ("representative_days.png",
         ("Figure 6. A winter and a summer weekday from the held-out weeks: demand, grid import, battery level and "
          "the Q-learning agent's actions. The dashed line is the oracle."), 5.4),
    ]),
    ("**What the learned policies look like", [
        ("policy_heatmap_selected.png",
         "Figure 7. Preferred action by battery level and hour for the selected (coarse-grid) Q-learning agent, seed 42.",
         5.6),
    ]),
]


def prepare_markdown(text: str) -> str:
    """Draft markdown -> pandoc input with metadata, figures and highlighted placeholders."""
    # Title block: H1 becomes metadata; the draft-only notes and the figure appendix are removed.
    title = re.search(r"^# (.+)$", text, re.MULTILINE).group(1)
    text = re.sub(r"^# .+\n+", "", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"\*Draft notes:.*?\*\n+", "", text, flags=re.DOTALL)
    text = re.sub(r"\n---\n+(?=\*\*|\n|##)", "\n", text, count=1)          # the rule after the draft notes
    text = text.split("## Appendix B.")[0].rstrip() + "\n"

    # Figures after the paragraphs they illustrate.
    blocks = text.split("\n\n")
    for prefix, figs in FIGURES:
        idx = next((i for i, b in enumerate(blocks) if b.startswith(prefix)), None)
        if idx is None:
            raise SystemExit(f"anchor not found in the draft: {prefix!r}")
        for name, _, _ in figs:
            if not (FIG / name).exists():
                raise SystemExit(f"missing figure: {name} (run scripts/make_report.py)")
        md = [f"![{cap}]({FIG / name}){{width={width:.1f}in}}" for name, cap, width in figs]
        blocks[idx + 1:idx + 1] = md
    text = "\n\n".join(blocks)

    # Yellow highlight for [YOU ...] placeholders (raw Word XML run).
    def highlight(m: re.Match) -> str:
        run = ('<w:r><w:rPr><w:b/><w:highlight w:val="yellow"/></w:rPr>'
               f'<w:t xml:space="preserve">{escape(m.group(0))}</w:t></w:r>')
        return f"`{run}`{{=openxml}}"

    text = re.sub(r"\[YOU[^\]]*\]", highlight, text)
    meta = f"---\ntitle: \"{title}\"\nauthor: \"{AUTHOR}\"\n---\n\n"
    return meta + text


def make_reference_docx(path: Path) -> None:
    """pandoc's default reference.docx with compact fonts, margins and bordered tables."""
    default = path.with_name("ref_default.docx")
    with open(default, "wb") as fh:
        fh.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                                check=True, capture_output=True).stdout)

    def patch_styles(s: str) -> str:
        s = re.sub(r'(<w:rPrDefault>.*?)<w:rFonts [^>]*/>', r'\1<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" '
                   r'w:eastAsia="Calibri" w:cs="Calibri" />', s, count=1, flags=re.DOTALL)
        s = s.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />', '<w:sz w:val="21" />\n        <w:szCs w:val="21" />', 1)
        s = s.replace('<w:spacing w:after="200" />', '<w:spacing w:after="100" />', 1)
        s = s.replace('<w:spacing w:before="180" w:after="180" />', '<w:spacing w:before="40" w:after="100" />', 1)
        # Compact (table cells, tight lists) and code a little smaller.
        s = re.sub(r'(w:styleId="Compact".*?</w:pPr>)', r'\1<w:rPr><w:sz w:val="19" /><w:szCs w:val="19" /></w:rPr>',
                   s, count=1, flags=re.DOTALL)
        s = re.sub(r'(w:styleId="VerbatimChar".*?)<w:sz w:val="22" />', r'\1<w:sz w:val="19" />', s, count=1, flags=re.DOTALL)
        # Bordered tables with a shaded bold header row.
        borders = "".join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="A6A6A6" />'
                          for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
        s = re.sub(r'(w:styleId="Table".*?<w:tblInd [^>]*/>)', rf'\1<w:tblBorders>{borders}</w:tblBorders>', s,
                   count=1, flags=re.DOTALL)
        # The reference style already has a first-row rule; replace it (one per type) with bold + shading.
        head = ('<w:tblStylePr w:type="firstRow"><w:rPr><w:b /></w:rPr><w:tcPr>'
                '<w:tcBorders><w:bottom w:val="single" w:sz="8" w:space="0" w:color="7F7F7F" /></w:tcBorders>'
                '<w:shd w:val="clear" w:color="auto" w:fill="E7EDF3" /><w:vAlign w:val="bottom" /></w:tcPr>'
                '</w:tblStylePr>')
        m = re.search(r'w:styleId="Table".*?</w:style>', s, re.DOTALL)
        block = re.sub(r'<w:tblStylePr w:type="firstRow">.*?</w:tblStylePr>', head, m.group(0), count=1, flags=re.DOTALL)
        return s[:m.start()] + block + s[m.end():]

    def patch_document(d: str) -> str:
        sect = ('<w:sectPr><w:footnotePr><w:numRestart w:val="eachSect" /></w:footnotePr>'
                '<w:pgSz w:w="12240" w:h="15840" /><w:pgMar w:top="1220" w:right="1220" w:bottom="1220" '
                'w:left="1220" w:header="720" w:footer="720" w:gutter="0" /></w:sectPr>')
        return re.sub(r"<w:sectPr>.*?</w:sectPr>", sect, d, count=1, flags=re.DOTALL)

    with zipfile.ZipFile(default) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                data = patch_styles(data.decode("utf8")).encode("utf8")
            elif item.filename == "word/document.xml":
                data = patch_document(data.decode("utf8")).encode("utf8")
            zout.writestr(item, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "Final_Report.docx"))
    args = parser.parse_args()
    if not shutil.which("pandoc"):
        raise SystemExit("pandoc is required: https://pandoc.org/installing.html")
    draft = (ROOT / "docs" / "REPORT_DRAFT.md").read_text(encoding="utf8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        md, ref = tmp / "report.md", tmp / "reference.docx"
        md.write_text(prepare_markdown(draft), encoding="utf8")
        make_reference_docx(ref)
        subprocess.run(["pandoc", str(md), "-f", "markdown-tex_math_dollars", "-t", "docx",
                        "--reference-doc", str(ref), "--resource-path", str(ROOT), "-o", args.out], check=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
