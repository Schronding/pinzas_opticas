import numpy as np
import pandas as pd
import os
import sys
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
# Carpeta solicitada: datos_experimental
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales') 

FILES = {
    'sx': os.path.join(DATA_DIR, 'datos_sx.dat'),
    'sy': os.path.join(DATA_DIR, 'datos_sy.dat'),
    'sum': os.path.join(DATA_DIR, 'datos_sum.dat'),
    'calib': os.path.join(DATA_DIR, 'datos_calibracion.txt')
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_imagenes')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def lorentzian(f, fc, D):
    return D / (np.pi**2 * (fc**2 + f**2))

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
    except Exception: pass
    return meta

def procesar_y_guardar():
    print(f"--- Iniciando Análisis desde: {DATA_DIR} ---")
    
    # 1. Cargar
    try:
        for key, path in FILES.items():
            if not os.path.exists(path):
                return {'error': f"Falta el archivo: {path}\nVerifica la carpeta 'datos_experimental'."}

        df_sx = pd.read_csv(FILES['sx'], sep='\t', header=None)
        df_sy = pd.read_csv(FILES['sy'], sep='\t', header=None)
        df_sum = pd.read_csv(FILES['sum'], sep='\t', header=None)
        meta = leer_metadatos(FILES['calib'])
        
    except Exception as e:
        return {'error': f"Error leyendo archivos: {e}"}

    # 2. Procesar
    raw_sx = df_sx.values.flatten()
    raw_sy = df_sy.values.flatten()
    raw_sum = df_sum.values.flatten()
    min_len = min(len(raw_sx), len(raw_sy), len(raw_sum))
    
    norm_x = (raw_sx[:min_len] / raw_sum[:min_len]) 
    norm_x = norm_x - np.mean(norm_x)
    norm_y = (raw_sy[:min_len] / raw_sum[:min_len])
    norm_y = norm_y - np.mean(norm_y)

    # 3. PSD
    fs = 20000 
    f_x, Pxx = signal.welch(norm_x, fs, nperseg=4096) # Más resolución en baja frecuencia
    f_y, Pyy = signal.welch(norm_y, fs, nperseg=4096)


    mask = (f_x > 1) & (f_x < 2000) 
    
 
    p0_x = [15, np.mean(Pxx[:10])]
    p0_y = [15, np.mean(Pyy[:10])]
    
    try:
        popt_x, _ = curve_fit(lorentzian, f_x[mask], Pxx[mask], p0=p0_x, bounds=(0, np.inf))
        popt_y, _ = curve_fit(lorentzian, f_y[mask], Pyy[mask], p0=p0_y, bounds=(0, np.inf))
    except Exception as e:
        return {'error': f"Fallo en el ajuste: {e}"}

    fc_x, D_x = popt_x
    fc_y, D_y = popt_y

    # 5. Física (Unidades Correctas)
    eta_agua = 0.00089 # Pa·s
    gamma = 6 * np.pi * eta_agua * meta['R']
    
    # k en N/m
    kx_Nm = 2 * np.pi * gamma * fc_x
    ky_Nm = 2 * np.pi * gamma * fc_y
    
    # Conversión a pN/um
    # 1 N/m = 1e6 pN/um
    kx_pN_um = kx_Nm * 1e6
    ky_pN_um = ky_Nm * 1e6

    # --- Gráfica ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Eje X
    ax1.loglog(f_x, Pxx, 'b.', alpha=0.2, label='Datos')
    ax1.loglog(f_x, lorentzian(f_x, *popt_x), 'k--', lw=2, label=f'Ajuste ($f_c$={fc_x:.2f} Hz)')
    ax1.axvline(x=fc_x, color='r', ls=':', alpha=0.8)
    ax1.set_title(f"Eje X: $k \\approx {kx_pN_um:.2f}$ pN/$\\mu$m")
    ax1.set_xlabel('Frecuencia (Hz)'); ax1.set_ylabel('PSD ($V^2/Hz$)')
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.4)

    # Eje Y
    ax2.loglog(f_y, Pyy, 'g.', alpha=0.2, label='Datos')
    ax2.loglog(f_y, lorentzian(f_y, *popt_y), 'k--', lw=2, label=f'Ajuste ($f_c$={fc_y:.2f} Hz)')
    ax2.axvline(x=fc_y, color='r', ls=':', alpha=0.8)
    ax2.set_title(f"Eje Y: $k \\approx {ky_pN_um:.2f}$ pN/$\\mu$m")
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_psd_completo.png'), dpi=150)
    
    return {
        'fig': fig,
        'traj_x': norm_x, 'traj_y': norm_y,
        'kx': kx_Nm, 'ky': ky_Nm, # Se quedan en SI para cálculos físicos si se necesitan
        'kx_display': kx_pN_um,   # Valor listo para mostrar
        'ky_display': ky_pN_um,
        'fc_x': fc_x, 'fc_y': fc_y
    }

if __name__ == '__main__':
    print("\n=== PRUEBA TERMINAL ===")
    res = procesar_y_guardar()
    if 'error' in res:
        print(f"❌ {res['error']}")
    else:
        print(f"✅ ÉXITO.")
        print(f"  fc_x: {res['fc_x']:.2f} Hz -> kx: {res['kx_display']:.2f} pN/um")
        print(f"  fc_y: {res['fc_y']:.2f} Hz -> ky: {res['ky_display']:.2f} pN/um")
        plt.show()