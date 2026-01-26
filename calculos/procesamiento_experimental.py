import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Configuration and paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_imagenes')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FILES = {
    'sx': os.path.join(DATA_DIR, 'datos_sx.dat'), 
    'sy': os.path.join(DATA_DIR, 'datos_sy.dat'),
    'calib': os.path.join(DATA_DIR, 'datos_calibracion.txt')
}

def lorentzian(f, fc, D):
    """
    Theoretical Lorentzian Power Spectral Density (PSD) for a trapped bead.
    S(f) = D / (pi^2 * (fc^2 + f^2))
    """
    return D / (np.pi**2 * (fc**2 + f**2))

def calculate_psd_fft(signal, fs):
    """
    Computes the One-Sided Power Spectral Density (PSD) using FFT.
    """
    N = len(signal)
    dt = 1.0 / fs
    
    # 1. Compute FFT (Fast Fourier Transform)
    # Returns complex numbers (Z = a + bi) containing Magnitude and Phase.
    fft_vals = np.fft.fft(signal)
    
    # 2. Compute Power (Magnitude Squared)
    # |Z|^2 = a^2 + b^2. We square the Magnitude (Amplitude), not the sine wave itself.
    # Physically, this converts Amplitude into Energy or Power.
    psd = (np.abs(fft_vals)**2) * dt / N

    # 3. Convert to One-Sided Spectrum
    # We multiply by 2 to satisfy Parseval's Theorem (Conservation of Energy).
    # Since we discard the negative half of the spectrum (which mirrors the positive half),
    # we must double the positive values to keep the total energy correct.
    psd = psd * 2 
    
    # 4. Get Frequency Bins
    # A 'bin center' is the specific frequency tested at that index.
    # Since we can't measure infinite frequencies, we chop the spectrum into discrete 'bins'.
    freqs = np.fft.fftfreq(N, dt)
    
    # 5. Mask: Keep only positive frequencies
    # We ignore negative values because they are just a mathematical mirror 
    # for real-valued signals. They don't represent "negative time," just a phasor 
    # rotating in the opposite direction.
    mask = (freqs > 0)
    
    return freqs[mask], psd[mask]

def read_metadata(filepath):
    """
    Reads temperature and radius from the calibration file.
    """
    meta = {'T': 298.15, 'R': 1.0e-6} 
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Temperature (K)" in line:
                parts = lines[i+1].strip().split('\t')
                if len(parts) > 3: meta['T'] = float(parts[3])
                if len(parts) > 4: meta['R'] = float(parts[4]) * 1e-6
    except Exception as e: 
        print(f"Warning reading metadata: {e}. Using defaults.")
    return meta

def process_and_save():
    print(f"--- Starting Analysis (Direct FFT Method) ---")
    
    try:
        # Load Data
        # Updated to 'sep' to avoid the FutureWarning
        # We use regex r'\s+' to handle variable spaces/tabs
        df_sx = pd.read_csv(FILES['sx'], sep=r'\s+', header=None)
        df_sy = pd.read_csv(FILES['sy'], sep=r'\s+', header=None)
        meta = read_metadata(FILES['calib'])
        
        # Handle 1-column vs 2-column data formats
        raw_sx = df_sx.iloc[:, 0].values if df_sx.shape[1] == 1 else df_sx.iloc[:, 1].values
        raw_sy = df_sy.iloc[:, 0].values if df_sy.shape[1] == 1 else df_sy.iloc[:, 1].values
        
    except Exception as e:
        return {'error': f"Error loading data: {e}.\nCheck files in '{DATA_DIR}'."}

    # Normalize Data (Zero Mean)
    norm_x = raw_sx - np.mean(raw_sx)
    norm_y = raw_sy - np.mean(raw_sy)

    # Calculate PSD
    fs = 20000 
    f_x, Pxx = calculate_psd_fft(norm_x, fs)
    f_y, Pyy = calculate_psd_fft(norm_y, fs)

    # Lorentzian Fitting
    mask_fit = (f_x > 2) & (f_x < 8000)
    
    p0_x = [50, np.mean(Pxx[(f_x > 2) & (f_x < 10)])]
    p0_y = [50, np.mean(Pyy[(f_y > 2) & (f_y < 10)])]
    
    try:
        # Added 'bounds' to prevent negative frequencies.
        # bounds=([lower_fc, lower_D], [upper_fc, upper_D])
        # We force fc and D to be positive (0 to infinity).
        popt_x, _ = curve_fit(lambda f, fc, D: np.log(lorentzian(f, fc, D)), 
                              f_x[mask_fit], np.log(Pxx[mask_fit]), 
                              p0=p0_x, bounds=(0, np.inf))
        
        popt_y, _ = curve_fit(lambda f, fc, D: np.log(lorentzian(f, fc, D)), 
                              f_y[mask_fit], np.log(Pyy[mask_fit]), 
                              p0=p0_y, bounds=(0, np.inf))
    except Exception as e:
        print(f"Warning: Curve fit failed ({e}). Using initial guess.")
        popt_x = p0_x
        popt_y = p0_y

    fc_x, D_x = popt_x
    fc_y, D_y = popt_y

    # Physics Calculation
    eta = 0.00089 
    gamma = 6 * np.pi * eta * meta['R']
    
    kx = 2 * np.pi * gamma * fc_x
    ky = 2 * np.pi * gamma * fc_y
    
    kx_disp = kx * 1e6
    ky_disp = ky * 1e6

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def plot_axis(ax, f, P, popt, k_val, label_axis):
        ax.loglog(f, P, color='royalblue', alpha=0.5, lw=0.5, label='Raw FFT Data')
        
        f_fit = np.logspace(np.log10(min(f)), np.log10(max(f)), 500)
        ax.loglog(f_fit, lorentzian(f_fit, *popt), 'r--', lw=2.5, 
                  label=f'Fit ($f_c$={popt[0]:.1f}Hz)')
        
        ax.axvline(x=popt[0], color='orange', linestyle=':', lw=2)
        
        ax.set_title(f"Axis {label_axis}: $k \\approx {k_val:.2f}$ pN/$\\mu$m")
        ax.set_xlabel('Frequency (Hz)')
        if label_axis == 'X': ax.set_ylabel('PSD ($V^2/Hz$)')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        ax.set_xlim(1, fs/2)

    plot_axis(ax1, f_x, Pxx, popt_x, kx_disp, 'X')
    plot_axis(ax2, f_y, Pyy, popt_y, ky_disp, 'Y')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'analisis_psd_fft.png')
    plt.savefig(plot_path, dpi=150)
    
    print(f"SUCCESS: fc_x={fc_x:.2f}Hz, fc_y={fc_y:.2f}Hz")
    
    return {
        'fig': fig,
        'traj_x': norm_x[::10], 
        'traj_y': norm_y[::10],
        'kx_display': kx_disp,
        'ky_display': ky_disp,
        'fc_x': fc_x, 'fc_y': fc_y
    }

if __name__ == '__main__':
    res = process_and_save()
    if 'error' in res:
        print(res['error'])
    else:
        plt.show()