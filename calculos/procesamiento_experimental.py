import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# By using the 'os' library we are able to more easily handle the 
# file structure, as we take the directory portion of a given file
# system path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales') 

# While the code was optimized for debian 12, by having use of 
# 'os.path.join' the library makes sure that all the paths are 
# ajusted to the specific operating system where the optical tweezers
# are being executed. 
FILES = {
    'sx': os.path.join(DATA_DIR, 'datos_sx.dat'), 
    'sy': os.path.join(DATA_DIR, 'datos_sy.dat'),
    'calib': os.path.join(DATA_DIR, 'datos_calibracion.txt')
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_imagenes')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def lorentzian(f, fc, D):
    """
    Theoretical Lorentzian behavior for brownian motion.
    S(f) = D / (pi^2 * (fc^2 + f^2))
    """
    return D / (np.pi**2 * (fc**2 + f**2))

def calcular_psd_fft(senal, fs):

    N = len(senal)
    # By dividing 1 by the frequency we're obtaining the specific
    # number of Hertz we need to have our time interval Delta t. 
    dt = 1.0 / fs
    
    # Fast Fourier Transform
    # Standard normalization of numpy to get the correct physical amplitudes. 
    fft_vals = np.fft.fft(senal)

    
    # Calculate potency (one-sided PSD). The reason why we use just 
    # one side is that... I don't really know. 
    # Formula: |FFT|^2 * dt / N

    psd = (np.abs(fft_vals)**2) * dt / N
    # As the 'ndarray' that is created by the 'np.fft.fft()' function
    # is being squared, does that mean that just the amplitude is being 
    # squared? Or are the sines and cosines too? I know they can, but
    # what would that mean physically? 

    psd = psd * 2 
    # We multiply by 2 to compensate the negative part of the 
    # spectrum. 

    freqs = np.fft.fftfreq(N, dt)
    # What is a 'bin center'? It seems that the only difference 
    # between ...fft and ...fftreq are those bin centers. Are these
    # the points where the cosine and sine frequencies of the discrete
    # fourier transform are equal to 1? 

    mask = (freqs > 0)
    # This seems to be a similar action than when we performed the 
    # PSD, completely ignoring the values at the left... which might
    # be negative time, and as negative time doesn't really help
    # in our simulation, as it would be what passed before, we can 
    # ignore it? 
    return freqs[mask], psd[mask]

def leer_metadatos(filepath):
    meta = {'T': 298.15, 'R': 1.0e-6} 
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Temperature (K)" in line:
                parts = lines[i+1].strip().split('\t')
                if len(parts) > 3: meta['T'] = float(parts[3])
                if len(parts) > 4: meta['R'] = float(parts[4]) * 1e-6
    except: pass
    return meta

def procesar_y_guardar():
    print(f"--- Iniciando Análisis (Método FFT Directa) ---")
    print(f"Directorio: {DATA_DIR}")
    
    try:
        df_sx = pd.read_csv(FILES['sx'], sep='\s+', header=None, engine='python')
        # Why we use the 'engine' python? This means that it is python
        # that process the values? I assume it is by the library, as it is
        # based on C++. That python read the values seems slow... but
        # these are just the data, python should be enough. 
        df_sy = pd.read_csv(FILES['sy'], sep='\s+', header=None, engine='python')
        meta = leer_metadatos(FILES['calib'])
        
        raw_sx = df_sx.iloc[:, 0].values if df_sx.shape[1] == 1 else df_sx.iloc[:, 1].values
        raw_sy = df_sy.iloc[:, 0].values if df_sy.shape[1] == 1 else df_sy.iloc[:, 1].values
        
    except Exception as e:
        return {'error': f"Error cargando datos: {e}.\nRevisa nombres en 'datos_experimentales'."}

    # This is the normalization of data that appeared on the 
    # 'Viscosity measurements on micron-size scale using optical tweezers'
    # paper. 
    norm_x = raw_sx - np.mean(raw_sx)
    norm_y = raw_sy - np.mean(raw_sy)

    # 3. Calcular PSD (FFT)
    fs = 20000 # Hz (Frecuencia de muestreo típica)
    f_x, Pxx = calcular_psd_fft(norm_x, fs)
    # I assume pxx and pyy are just the power spectral densities in the
    # x and y axis. 
    f_y, Pyy = calcular_psd_fft(norm_y, fs)

    # 4. Lorentzian adjustment (Fitting)
    # We block the very high frequencies, which normally are just 
    # pure noise
    # We also block the frequencies that are very low, which are 
    # normally a minor drift of the laser. 
    mask_fit = (f_x > 2) & (f_x < 8000)
    
    # Initial guess (p0):
    # A corner frequency of 50 Hz is usually a middle, conservative
    # number 
    # D = Mean of the plateau (meseta) at low frequency
    # Why is the plateau important? On the knee of the frequencies
    # I do see how it remains fairly stable up until it drops down. 
    # I suppose we do this to know where the drop occurs numerically. 
    p0_x = [50, np.mean(Pxx[(f_x > 2) & (f_x < 10)])]
    p0_y = [50, np.mean(Pyy[(f_y > 2) & (f_y < 10)])]
    
    try:
        # We adjust Log(PSD) to give equal weight the knee and the tail. 
        # This helps a lot to see the knee. 
        popt_x, _ = curve_fit(lambda f, fc, D: np.log(lorentzian(f, fc, D)), 
                              f_x[mask_fit], np.log(Pxx[mask_fit]), p0=p0_x)
        
        popt_y, _ = curve_fit(lambda f, fc, D: np.log(lorentzian(f, fc, D)), 
                              f_y[mask_fit], np.log(Pyy[mask_fit]), p0=p0_y)
    except Exception as e:
        print(f"Warning: Ajuste falló ({e}), usando valores por defecto.")
        popt_x = p0_x
        popt_y = p0_y

    fc_x, D_x = popt_x
    fc_y, D_y = popt_y

    # 5. Física (Cálculo de k)
    eta = 0.00089 # Pa·s (Agua)
    gamma = 6 * np.pi * eta * meta['R']
    
    # Rigidez k = 2*pi*gamma*fc
    kx = 2 * np.pi * gamma * fc_x
    ky = 2 * np.pi * gamma * fc_y
    
    # Convertir a pN/um para mostrar (x 1e6)
    kx_disp = kx * 1e6
    ky_disp = ky * 1e6

    # --- GRÁFICA (REPRODUCCIÓN DE REFERENCIA) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Función auxiliar para plotear bonito
    def plot_axis(ax, f, P, popt, k_val, label_axis):
        ax.loglog(f, P, color='royalblue', alpha=0.5, lw=0.5, label='Datos FFT (Crudos)')
        
        # Línea de ajuste
        f_fit = np.logspace(np.log10(min(f)), np.log10(max(f)), 500)
        ax.loglog(f_fit, lorentzian(f_fit, *popt), 'r--', lw=2.5, label=f'Ajuste ($f_c$={popt[0]:.1f}Hz)')
        
        # Línea vertical en la rodilla
        ax.axvline(x=popt[0], color='orange', linestyle=':', lw=2)
        
        ax.set_title(f"Eje {label_axis}: $k \\approx {k_val:.2f}$ pN/$\\mu$m")
        ax.set_xlabel('Frecuencia (Hz)')
        if label_axis == 'X': ax.set_ylabel('PSD ($V^2/Hz$)')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        
        # Limites para que se vea "estilo paper"
        ax.set_xlim(1, fs/2)

    plot_axis(ax1, f_x, Pxx, popt_x, kx_disp, 'X')
    plot_axis(ax2, f_y, Pyy, popt_y, ky_disp, 'Y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_psd_fft.png'), dpi=150)
    
    print(f"✅ ÉXITO: fc_x={fc_x:.2f}Hz, fc_y={fc_y:.2f}Hz")
    
    return {
        'fig': fig,
        'traj_x': norm_x[::10], # Diezmamos para que la animación sea ligera
        'traj_y': norm_y[::10],
        'kx_display': kx_disp,
        'ky_display': ky_disp,
        'fc_x': fc_x, 'fc_y': fc_y
    }

if __name__ == '__main__':
    res = procesar_y_guardar()
    if 'error' in res:
        print(res['error'])
    else:
        plt.show()