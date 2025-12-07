import os
import csv
import subprocess
import shutil
from collections import defaultdict

SUMMARY_DIR = "summaries"
SUMMARY_CSV = os.path.join(SUMMARY_DIR, "pupil_hht_summary.csv")
REPORT_MD = os.path.join(SUMMARY_DIR, "summary_report.md")
REPORT_PDF = os.path.join(SUMMARY_DIR, "summary_report.pdf")


def load_summary():
    rows = []
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def escape_markdown(text: str) -> str:
    """
    Escape characters that can upset LaTeX/Pandoc or Markdown.
    We mainly care about underscores here.
    """
    return text.replace("_", r"\_")


def write_markdown_report(rows):
    # Group by stimulus_label
    by_stim = defaultdict(list)
    for row in rows:
        by_stim[row["stimulus_label"]].append(row)

    # Map full file_tag -> short ID (F1, F2, ...)
    file_id_map = {}
    next_id = 1

    lines = []

    # YAML header for Pandoc/LaTeX
    # - landscape so tables fit more comfortably
    # - smaller font to help width
    lines.append("---")
    lines.append("geometry: margin=1in, landscape")
    lines.append("fontsize: 9pt")
    lines.append("---\n")

    lines.append("# Pupil HHT Summary Report\n")

    # Build the main tables using short IDs
    for stim, stim_rows in sorted(by_stim.items()):
        lines.append(f"## Stimulus: {stim}\n")
        lines.append("")

        # Table header: note file_id instead of file_tag
        lines.append("| file_id | fs (Hz) | duration (s) | dom. band | dom. power | VLF | LF | MF | HF |")
        lines.append("|---------|--------:|------------:|-----------|-----------:|----:|---:|---:|---:|")

        for r in stim_rows:
            file_tag = r["file_tag"]
            if file_tag not in file_id_map:
                file_id_map[file_tag] = f"F{next_id}"
                next_id += 1

            file_id = file_id_map[file_tag]

            lines.append(
                "| {file_id} | {fs_hz:.1f} | {dur:.3f} | {dband} | {dpow:.4f} | "
                "{VLF:.4f} | {LF:.4f} | {MF:.4f} | {HF:.4f} |".format(
                    file_id=file_id,
                    fs_hz=float(r["fs_hz"]),
                    dur=float(r["post_duration_s"]),
                    dband=r["dominant_band"],
                    dpow=float(r["dominant_band_power"]),
                    VLF=float(r["VLF_power"]),
                    LF=float(r["LF_power"]),
                    MF=float(r["MF_power"]),
                    HF=float(r["HF_power"]),
                )
            )
        lines.append("")  # blank line after each table

    # Add a reference section with the full filenames
    lines.append("## File tag reference\n")
    lines.append("")
    # Sort by file_id (F1, F2, ...) for stable ordering
    for file_tag, file_id in sorted(file_id_map.items(), key=lambda x: int(x[1][1:])):
        lines.append(f"- **{file_id}**: {escape_markdown(file_tag)}")

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote Markdown report to {REPORT_MD}")


def write_pdf_with_pandoc():
    """Use Pandoc + xelatex to convert the Markdown report to PDF."""
    pandoc_path = shutil.which("pandoc")
    if pandoc_path is None:
        print(
            "Pandoc is not found in PATH. Install it or add it to PATH to get the PDF.\n"
            "Skipping PDF generation."
        )
        return

    cmd = [
        pandoc_path,
        REPORT_MD,
        "-o",
        REPORT_PDF,
        "--pdf-engine=xelatex",
    ]

    print("Running:", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Pandoc failed with return code:", e.returncode)
        if e.stdout:
            print("stdout:\n", e.stdout)
        if e.stderr:
            print("stderr:\n", e.stderr)
        return

    print(f"Wrote PDF report to {REPORT_PDF}")


def main():
    rows = load_summary()
    write_markdown_report(rows)
    write_pdf_with_pandoc()


if __name__ == "__main__":
    main()
