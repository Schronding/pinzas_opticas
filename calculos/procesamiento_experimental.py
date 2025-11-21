# -*- coding: utf-8 -*-
"""
Módulo de procesamiento de datos experimentales.
Versión alineada con 'Analisis_señal_profe.ipynb':
- Usa FFT directa (no Welch) para máxima resolución en baja frecuencia.
- Elimina la doble normalización (los archivos _nor ya vienen listos).
- Implementa un ajuste robusto para encontrar la frecuencia de corte (fc).
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales') # Carpeta corregida

FILES = {
    # Asumimos que renombraste tus archivos _nor.dat a estos nombres estandarizados
    # O si prefieres, el script puede leer cualquier .dat que encuentre.
    'sx': os.path.join(DATA_DIR, 'datos_sx.dat'), 
    'sy': os.path.join(DATA_DIR, 'datos_sy.dat'),
    'calib': os.path.join(DATA_DIR, 'datos_calibracion.txt')
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_imagenes')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def lorentzian(f, fc, D):
    """
    Modelo Lorentziano teórico para movimiento Browniano.
    S(f) = D / (pi^2 * (fc^2 + f^2))
    """
    return D / (np.pi**2 * (fc**2 + f**2))

def calcular_psd_fft(senal, fs):
    """
    Calcula el PSD usando FFT directa (Estilo del Notebook del Profe).
    Mantiene la resolución completa en bajas frecuencias.
    """
    N = len(senal)
    dt = 1.0 / fs
    
    # 1. FFT
    # Normalización estándar de numpy para obtener amplitudes físicas correctas
    fft_vals = np.fft.fft(senal)
    
    # 2. Calcular Potencia (One-Sided PSD)
    # Formula: |FFT|^2 * dt / N
    # Multiplicamos por 2 para compensar la parte negativa del espectro que borramos
    psd = (np.abs(fft_vals)**2) * dt / N
    psd = psd * 2 
    
    # 3. Frecuencias
    freqs = np.fft.fftfreq(N, dt)
    
    # 4. Filtrar solo frecuencias positivas
    # Omitimos la componente DC (índice 0) y las negativas
    mask = (freqs > 0)
    return freqs[mask], psd[mask]

def leer_metadatos(filepath):
    meta = {'T': 298.15, 'R': 1.0e-6} # Default: 25°C, 1 micra radio
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
    
    # 1. Cargar Archivos
    try:
        # Intentamos cargar con pandas, asumiendo tabuladores o espacios
        df_sx = pd.read_csv(FILES['sx'], sep='\s+', header=None, engine='python')
        df_sy = pd.read_csv(FILES['sy'], sep='\s+', header=None, engine='python')
        meta = leer_metadatos(FILES['calib'])
        
        # Si el archivo tiene una sola columna, la tomamos directa. 
        # Si tiene tiempo y señal, tomamos la señal (asumiendo col 1)
        raw_sx = df_sx.iloc[:, 0].values if df_sx.shape[1] == 1 else df_sx.iloc[:, 1].values
        raw_sy = df_sy.iloc[:, 0].values if df_sy.shape[1] == 1 else df_sy.iloc[:, 1].values
        
    except Exception as e:
        return {'error': f"Error cargando datos: {e}.\nRevisa nombres en 'datos_experimentales'."}

    # 2. Pre-procesamiento (SOLO CENTRAR)
    # ¡No dividimos por la suma! Los datos _nor ya vienen normalizados en amplitud.
    norm_x = raw_sx - np.mean(raw_sx)
    norm_y = raw_sy - np.mean(raw_sy)

    # 3. Calcular PSD (FFT)
    fs = 20000 # Hz (Frecuencia de muestreo típica)
    f_x, Pxx = calcular_psd_fft(norm_x, fs)
    f_y, Pyy = calcular_psd_fft(norm_y, fs)

    # 4. Ajuste Lorentziano (Fitting)
    # Bloqueamos frecuencias muy altas (>8000 Hz) que suelen ser puro ruido electrónico
    # Bloqueamos frecuencias muy bajas (<2 Hz) que suelen ser drift del láser
    mask_fit = (f_x > 2) & (f_x < 8000)
    
    # Adivinanza inicial (p0):
    # fc = 50 Hz (Un valor intermedio seguro)
    # D = Promedio de la meseta de baja frecuencia
    p0_x = [50, np.mean(Pxx[(f_x > 2) & (f_x < 10)])]
    p0_y = [50, np.mean(Pyy[(f_y > 2) & (f_y < 10)])]
    
    try:
        # Ajustamos LOG(PSD) para dar igual peso a la rodilla que a la cola
        # Esto ayuda MUCHÍSIMO a ver la rodilla.
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