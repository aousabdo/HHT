import os
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

PLOT_DIR = "plots"
OUTPUT_PDF = os.path.join(PLOT_DIR, "combined_plots.pdf")

def main():
    pngs = sorted([f for f in os.listdir(PLOT_DIR) if f.endswith(".png")])
    if not pngs:
        print("No PNG files found in plots/")
        return

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    for png in pngs:
        path = os.path.join(PLOT_DIR, png)

        pdf.add_page()

        # Title
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt=png, ln=True, align="L")

        # Load image to get dimensions
        img = Image.open(path)
        width, height = img.size
        aspect = height / width

        # Fit image to A4 width (190 mm)
        display_width = 190
        display_height = display_width * aspect

        # If image is too tall, scale down further
        max_height = 250
        if display_height > max_height:
            scale = max_height / display_height
            display_width *= scale
            display_height *= scale

        pdf.image(path, x=10, y=None, w=display_width, h=display_height)

    pdf.output(OUTPUT_PDF)
    print(f"PDF created: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
