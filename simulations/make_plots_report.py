import os
from collections import defaultdict

from PIL import Image
from fpdf import FPDF

PLOT_DIR = "plots"
OUTPUT_PDF = os.path.join(PLOT_DIR, "combined_plots_grouped.pdf")

# Prefixes we care about and how we group them
PREFIXES = [
    "imfs_",
    "event_locked_imfs_",
    "hilbert_spectrum_",
    "hht_3d_",
    "reconstruction_",
]


def classify_file(filename):
    """Return (kind, base) if filename matches one of our prefixes, else (None, None)."""
    if not filename.lower().endswith(".png"):
        return None, None
    for prefix in PREFIXES:
        if filename.startswith(prefix):
            base = filename[len(prefix):-4]  # strip prefix and .png
            return prefix, base
    return None, None


def group_plots():
    """
    Return a dict:
      base -> { kind_prefix : filepath }
    """
    groups = defaultdict(dict)
    for fname in os.listdir(PLOT_DIR):
        kind, base = classify_file(fname)
        if kind is None:
            continue
        path = os.path.join(PLOT_DIR, fname)
        groups[base][kind] = path
    return groups


def add_images_page(pdf, images, title):
    """
    Add a new page to the PDF with a title and one or more images
    stacked vertically, auto-scaled to fit.
    """
    if not images:
        return

    # Ensure title is Latin-1 safe for classic fpdf
    title = title.encode("latin-1", "replace").decode("latin-1")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, txt=title, ln=True, align="L")

    # Page geometry
    margin = 10
    page_w = pdf.w
    page_h = pdf.h
    max_width = page_w - 2 * margin

    # Available vertical space for images
    y = pdf.get_y() + 2
    available_height = page_h - margin - y
    n = len(images)
    spacing = 5
    total_spacing = spacing * (n - 1) if n > 1 else 0
    per_image_height = (available_height - total_spacing) / n

    for img_path in images:
        # Get image aspect ratio
        with Image.open(img_path) as im:
            w_px, h_px = im.size
        aspect = h_px / w_px

        # Constrain by height first
        target_h = per_image_height
        target_w = target_h / aspect

        # Also constrain by max_width
        if target_w > max_width:
            target_w = max_width
            target_h = target_w * aspect

        # Center horizontally
        x = margin + (max_width - target_w) / 2

        pdf.image(img_path, x=x, y=y, w=target_w, h=target_h)

        y += target_h + spacing


def main():
    groups = group_plots()
    if not groups:
        print("No grouped PNGs found in plots/.")
        return

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False, margin=10)

    for base in sorted(groups.keys()):
        files = groups[base]

        # IMF page: imfs_ + event_locked_imfs
        imf_imgs = []
        if "imfs_" in files:
            imf_imgs.append(files["imfs_"])
        if "event_locked_imfs_" in files:
            imf_imgs.append(files["event_locked_imfs_"])
        if imf_imgs:
            add_images_page(pdf, imf_imgs, f"{base} - IMF plots")

        # Spectrum page: hilbert_spectrum + hht_3d
        spec_imgs = []
        if "hilbert_spectrum_" in files:
            spec_imgs.append(files["hilbert_spectrum_"])
        if "hht_3d_" in files:
            spec_imgs.append(files["hht_3d_"])
        if spec_imgs:
            add_images_page(pdf, spec_imgs, f"{base} - Hilbert / HHT spectra")

        # Reconstruction page
        recon_imgs = []
        if "reconstruction_" in files:
            recon_imgs.append(files["reconstruction_"])
        if recon_imgs:
            add_images_page(pdf, recon_imgs, f"{base} - Reconstruction")

    pdf.output(OUTPUT_PDF)
    print(f"Created grouped PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
