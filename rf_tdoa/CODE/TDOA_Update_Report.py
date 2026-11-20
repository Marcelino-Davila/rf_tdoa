import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'TDOA Pipeline System Improvements & Performance Update', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(5)

def create_pdf():
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # --- Section 1 ---
    pdf.chapter_title('1. Executive Summary')
    pdf.chapter_body(
        "Recent upgrades to the Time Difference of Arrival (TDOA) system architecture have "
        "significantly improved both detection reliability and localization precision. By integrating "
        "better hardware synchronization, increasing our sampling rate, and implementing advanced "
        "sub-sample digital signal processing, we have drastically reduced our error margins and "
        "improved our Signal-to-Noise Ratio (SNR)."
    )

    # --- Section 2 ---
    pdf.chapter_title('2. Key System Upgrades')
    pdf.chapter_body(
        "- Hardware Synchronization & Signal Conditioning: We remade the matched filter and noise "
        "covariance matrix using data captured with the GPSDO (GPS Disciplined Oscillator) and inline "
        "attenuators. This ensures tight clock synchronization across receivers and prevents front-end clipping.\n"
        "- Increased Sampling Rate (20 MHz): We increased the system sample rate to 20 MHz, improving "
        "our baseline integer time resolution to approx. 15 meters per sample.\n"
        "- Subsample Parabolic Interpolation: We upgraded the software pipeline to calculate fractional delays. "
        "Instead of snapping to the nearest integer sample, the cross-correlator fits a parabola to the "
        "correlation peak to find the true mathematical maximum. This provides a 10x to 20x improvement "
        "in resolution, bringing theoretical precision down to roughly 1-3 meters."
    )

    # --- Section 3 ---
    pdf.chapter_title('3. Signal Processing Pipeline Architecture')
    pdf.set_font('Courier', '', 9)
    pipeline_text = (
        "[Raw IQ Data @ 20 MHz]\n"
        "       |\n"
        "       +--> PATH 1: DETECTION (WHEN & WHETHER)\n"
        "       |      |\n"
        "       |      +--> Matched Filter (GPSDO-calibrated optimal filters)\n"
        "       |      +--> Magnitude Calculation\n"
        "       |      +--> Adaptive Thresholding (Mean + 3 sigma)\n"
        "       |             +--> Output: Confirms signal presence & rough timing.\n"
        "       |\n"
        "       +--> PATH 2: ESTIMATION (PRECISE TDOA)\n"
        "              |\n"
        "              +--> Complex Bandpass Filter (Isolates 50 kHz target tone)\n"
        "              +--> GCC-PHAT Cross-Correlation (Finds integer sample delay)\n"
        "              +--> Subsample Parabolic Interpolation (Calculates fractional delay)\n"
        "              +--> Output: High-Precision TDOA (Time & Distance in meters)"
    )
    pdf.multi_cell(0, 5, pipeline_text)
    pdf.ln(5)

    # --- Section 4 ---
    pdf.set_font('Arial', '', 11)
    pdf.chapter_title('4. Results & Performance Metrics')
    pdf.chapter_body(
        "The attached plots demonstrate the effectiveness of these improvements:\n\n"
        "- Clean Signal Isolation (Plot 1): The filter effectively isolates the target signal from the "
        "noise floor, providing a clean feed for the cross-correlator.\n"
        "- Robust Detection (Plot 2): The matched filters trigger well above the dynamic noise thresholds, "
        "confirming reliable active pulse detection.\n"
        "- Sub-Microsecond Precision (Plot 3): The GCC-PHAT outputs show extreme precision, successfully "
        "resolving fractional sub-sample offsets.\n"
        "- High Confidence/PSLR (Plot 4): Peak-to-Sidelobe Ratio (PSLR) cleared the 6 dB minimum usable "
        "threshold easily, achieving up to 21.8 dB. This proves the GPSDO and attenuators are working perfectly."
    )

    # --- Add Images ---
    images = [
        "plot1_raw_vs_bandpass.png", 
        "plot2_mf_detection.png", 
        "plot3_gcc_phat.png", 
        "plot4_pslr.png"
    ]
    
    for img in images:
        if os.path.exists(img):
            pdf.add_page()
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, f'Attached Figure: {img}', 0, 1, 'C')
            # Width of 190 keeps it inside standard A4 margins
            pdf.image(img, x=10, w=190)
        else:
            print(f"Warning: Could not find {img} in the current directory.")

    pdf.output('TDOA_Update_Report.pdf')
    print("Success! 'TDOA_Update_Report.pdf' has been generated.")

if __name__ == "__main__":
    create_pdf()