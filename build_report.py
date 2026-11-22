#!/usr/bin/env python3
"""Build the PDF report: runs the six be_0*.py programs, captures their output and figures.

Usage:  python build_report.py [--author "Name"] [--out Bioequivalence_Report_Python.pdf]
Requires: numpy scipy scikit-image matplotlib pillow reportlab
"""
import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = Path(__file__).parent
TEAL, DARK, GRAY = colors.HexColor("#0F6E56"), colors.HexColor("#2C2C2A"), colors.HexColor("#888780")

SECTIONS = [
    dict(file="be_01_mdrs_particle_analysis.py", fig="fig_01_mdrs.png",
         title="1. Morphologically-Directed Raman Spectroscopy (MDRS) for Nasal Product Bioequivalence",
         overview="MDRS combines automated particle imaging with Raman spectroscopy to characterise particle size, "
                  "shape and chemical composition in nasal sprays and inhaled products. This supports precise "
                  "comparison of generic and reference active pharmaceutical ingredients (APIs).",
         apps=["Image processing of particle images", "Quantification of particle size distributions and morphology parameters",
               "Statistical analysis of particle data"],
         remarks="This workflow can be extended by incorporating Raman spectral data for chemical identification, "
                 "integrating batch processing for multiple samples, and exporting results for regulatory reports.",
         notes="The original outline loads <i>particle_image.tif</i>. Because that file is not available, the program "
               "generates a synthetic micrograph (dark, slightly elongated particles on an unevenly lit background) so it "
               "runs as-is; pass your own image as the first argument to analyse real data. Adaptive thresholding "
               "(<i>threshold_local</i>) approximates MATLAB's <i>imbinarize</i>, so values will differ slightly. "
               "Circularity is clipped to 1 because pixel-based perimeters can overestimate it for tiny particles. "
               "No Raman spectra are included; chemical identification would need instrument spectra."),
    dict(file="be_02_dissolution_fit.py", fig="fig_02_dissolution.png",
         title="2. Nasal Spray API Particle Size by Dissolution Testing",
         overview="Particle size affects dissolution rate and drug bioavailability. Computational tools are used to model "
                  "dissolution kinetics, fit nonlinear models and simulate dissolution profiles.",
         apps=["Curve fitting dissolution data", "Modelling drug release kinetics (e.g. Weibull, Higuchi models)",
               "Predictive simulations linking particle size to dissolution"],
         remarks="This model can be linked to particle size distributions from MDRS data to predict dissolution changes "
                 "for different formulations.",
         notes="The Weibull fit uses <i>scipy.optimize.curve_fit</i> in place of MATLAB's <i>fminsearch</i> (same "
               "least-squares objective). A Higuchi fit is added for comparison; it fits poorly here because the data "
               "approach a plateau. The particle-size panel assumes the Weibull scale parameter is proportional to "
               "particle diameter. This is an illustrative assumption, not a validated relationship."),
    dict(file="be_03_ivrt_comparison.py", fig="fig_03_ivrt.png",
         title="3. Dissolution / In-Vitro Release Testing (IVRT) for Inhalation Products",
         overview="IVRT provides quantitative release rate data to compare generic and reference products and supports "
                  "in-vitro/in-vivo correlation (IVIVC).",
         apps=["Design and analysis of IVRT experiments", "Modelling release kinetics", "Statistical comparisons of release rates"],
         remarks="IVRT data can be correlated with pharmacokinetics using IVIVC models, improving bioequivalence predictions.",
         notes="The simple two-sample t-test from the original outline is kept, but it treats the six time points as "
               "independent replicates, which they are not, so its p-value should not be used as evidence of similarity. "
               "The f2 similarity factor (f2 of 50 or more is conventionally taken as similar) is added as the more "
               "usual comparison. With real data, use replicate vessels (typically 12) per product."),
    dict(file="be_04_breath_profile.py", fig="fig_04_breath.png",
         title="4. Patient Breath Profiling and Simulation",
         overview="Realistic breath pattern modelling enables simulation of drug aerosol deposition and pharmacokinetics.",
         apps=["Signal processing of spirometry data", "Simulation and visualisation of inhalation waveforms",
               "Integration with deposition models"],
         remarks="Real patient data can be imported and processed similarly; breath profiles can be coupled to aerosol "
                 "deposition CFD models.",
         notes="The simulated half-sine waveform is exactly as in the outline (12 breaths per minute, so one 5 s "
               "cycle with about 2.5 s of inhalation). A noisy copy is also smoothed with a Savitzky-Golay filter and "
               "re-analysed to show how measured spirometry would be processed. Flow units are arbitrary."),
    dict(file="be_05_lung_deposition.py", fig="fig_05_lung.png",
         title="5. Lung Regional Deposition Modelling",
         overview="Simulating aerosol deposition across lung regions helps predict local drug delivery and systemic absorption.",
         apps=["Solving differential equations that model particle transport and deposition",
               "Using numerical solvers for multi-compartment models"],
         remarks="More detailed anatomical and particle physics models can be incorporated using CFD data or multi-scale models.",
         notes="<i>solve_ivp</i> replaces MATLAB's <i>ode45</i>. Because this two-compartment system is linear, the "
               "program also checks the numerical result against the closed-form solution; the two agree to about 1e-10. "
               "The rate constants are illustrative, not fitted to any product."),
    dict(file="be_06_pbpk.py", fig="fig_06_pbpk.png",
         title="6. Physiologically Based Pharmacokinetic (PBPK) Modelling",
         overview="PBPK models integrate drug absorption, distribution, metabolism and excretion, predicting plasma "
                  "profiles from in-vitro and in-silico data.",
         apps=["Constructing PBPK models (MATLAB: SimBiology; here: a compartment ODE system)",
               "Parameter estimation and simulation of drug plasma concentration-time profiles"],
         remarks="This basic PBPK example can be expanded to include multiple organs, metabolic pathways, and integration "
                 "with deposition models.",
         notes="The SimBiology outline was not executed; the Python program implements the same two equations "
               "(lung to plasma absorption, plasma elimination) with <i>solve_ivp</i>, and adds concentrations (amount "
               "divided by compartment volume), Cmax/Tmax, AUC and a sensitivity run on the absorption rate. This is a "
               "minimal two-compartment model, not a multi-organ PBPK model."),
]


def build(author, out):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5,
                          alignment=TA_JUSTIFY, spaceAfter=6, textColor=DARK)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13.5, leading=17,
                        textColor=TEAL, spaceBefore=4, spaceAfter=6, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, textColor=DARK,
                        spaceBefore=8, spaceAfter=3, keepWithNext=1)
    title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27,
                           textColor=TEAL, alignment=0, spaceAfter=8)
    sub = ParagraphStyle("s", parent=body, fontSize=11.5, leading=15, textColor=GRAY, alignment=0)
    h2n = ParagraphStyle("h2n", parent=h2, keepWithNext=0)
    bullet = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=2, alignment=0)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.5, textColor=GRAY, alignment=0)
    note = ParagraphStyle("n", parent=body, fontSize=9, leading=12.5, backColor=colors.HexColor("#E1F5EE"),
                          borderPadding=6, spaceBefore=6, spaceAfter=10, alignment=0)
    code = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.2, textColor=DARK,
                          backColor=colors.HexColor("#F1EFE8"), borderPadding=5, spaceAfter=8)
    outp = ParagraphStyle("out", parent=code, backColor=colors.HexColor("#2C2C2A"), textColor=colors.HexColor("#E1F5EE"))

    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(GRAY)
        c.drawString(2 * cm, 1.2 * cm, "Bioequivalence testing of nasal and inhaled products - Python examples")
        c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        c.restoreState()

    def outbox(text):
        t = Table([[Preformatted(text, outp)]], colWidths=[17 * cm], hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), DARK), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        return t

    def figure(path, width_cm):
        w, h = Image.open(path).size
        return RLImage(str(path), width=width_cm * cm, height=width_cm * cm * h / w)

    st = [Spacer(1, 1.2 * cm),
          Paragraph("Advances in Bioequivalence Testing of Generic Nasal Spray and Inhaled Drug Products", title),
          Paragraph("with Python implementation examples", sub)]
    if author:
        st.append(Paragraph(author, sub))
    st += [Spacer(1, 0.5 * cm), Paragraph("Introduction", h1),
           Paragraph("Demonstrating bioequivalence (BE) between generic and reference orally inhaled and nasal drug "
                     "products (OINDPs) is crucial for regulatory approval. Innovations in analytical and computational "
                     "methods enhance the accuracy, efficiency and mechanistic understanding of BE studies. This report "
                     "walks through six such methods and gives a ready-to-run Python program for each, with its actual "
                     "output and figures.", body),
           Paragraph("How to use this report", h2),
           Paragraph("Each section has an overview, the Python program, the output it produced, a figure and a note on how "
                     "the program differs from the original MATLAB outline. Install the dependencies with "
                     "<font face='Courier'>pip install numpy scipy scikit-image matplotlib pillow reportlab</font>, run any "
                     "program with <font face='Courier'>python be_0N_*.py</font>, and rebuild this PDF with "
                     "<font face='Courier'>python build_report.py</font>.", body),
           Paragraph("<b>Important:</b> all data in these programs are illustrative or synthetic. They demonstrate "
                     "methods and are not validated for regulatory submissions without your own data, validation and "
                     "review.", note)]

    for s in SECTIONS:
        path = HERE / s["file"]
        res = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=HERE)
        if res.returncode != 0:
            raise SystemExit(f"{s['file']} failed:\n{res.stderr}")
        st += [PageBreak(), Paragraph(s["title"], h1), Paragraph("Overview", h2), Paragraph(s["overview"], body),
               Paragraph("Python applications", h2)]
        st += [Paragraph(a, bullet, bulletText="\u2022") for a in s["apps"]]
        st += [Paragraph(f"Program: {s['file']}", h2n), Preformatted(path.read_text().rstrip(), code),
               KeepTogether([Paragraph("Output", h2), outbox(res.stdout.rstrip()), Spacer(1, 8),
                             figure(HERE / "figures" / s["fig"], 15.5 if s["fig"] != "fig_01_mdrs.png" else 16.5),
                             Paragraph(f"Figure {s['title'][0]}. Figure produced by {s['file']}.", cap)]),
               Paragraph("Remarks", h2), Paragraph(s["remarks"], body),
               Paragraph("<b>Notes on this Python version:</b> " + s["notes"], note)]

    st += [PageBreak(), Paragraph("Summary", h1),
           Paragraph("This report demonstrates how computational tools can support advances in bioequivalence testing "
                     "for generic nasal and inhaled drug products by enabling:", body)]
    st += [Paragraph(t, bullet, bulletText="\u2022") for t in [
        "Automated image analysis of particle data (MDRS).", "Modelling and simulation of dissolution and release kinetics.",
        "Processing and simulation of patient breath profiles.", "Numerical modelling of lung deposition.",
        "Construction and simulation of PBPK pharmacokinetic models."]]
    st.append(Paragraph("Python's scientific stack (NumPy, SciPy, scikit-image, Matplotlib) supports reproducible, "
                        "scriptable analyses for each of these steps, which can be extended with real instrument and "
                        "study data.", body))
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                      bottomMargin=2 * cm, title="Bioequivalence Testing - Python Examples",
                      author=author or "").build(st, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="")
    ap.add_argument("--out", default=str(HERE / "Bioequivalence_Report_Python.pdf"))
    a = ap.parse_args()
    build(a.author, a.out)
    print("wrote", a.out)
