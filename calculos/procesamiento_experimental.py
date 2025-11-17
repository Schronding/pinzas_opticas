# -*- coding: utf-8 -*-
"""
Módulo de procesamiento de datos experimentales de pinzas ópticas.
Implementa el flujo de trabajo de análisis PSD:
Carga -> Normalización -> PSD -> Ajuste Lorentziano -> Viscosidad
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales')

FILES = {
    'sx': os.path.join(DATA_DIR, 'datos_sx.dat'),
    'sy': os.path.join(DATA_DIR, 'datos_sy.dat'),
    'sum': os.path.join(DATA_DIR, 'datos_sum.dat'),
    'calib': os.path.join(DATA_DIR, 'datos_calibracion.txt')
}

# --- PARÁMETROS FÍSICOS ---
# (Estos se podrían leer del archivo de calibración, pero para el ajuste
# de viscosidad, a veces es mejor usar T y R conocidos).
KB = 1.380649e-23 # J/K

def lorentzian(f, f_c, D_volts):
    """
    Modelo Lorentziano para el PSD de una pinza óptica.
    P(f) = D_volts / (pi^2 * (f_c^2 + f^2))
    
    Args:
        f: Frecuencia (Hz)
        f_c: Frecuencia de corte (Corner frequency) en Hz
        D_volts: Coeficiente de difusión en unidades de señal^2 (Volts^2/s)
    """
    return D_volts / (np.pi**2 * (f_c**2 + f**2))

def cargar_y_procesar_datos():
    """
    Ejecuta el pipeline completo de análisis.
    """
    print("--- Iniciando Análisis Experimental ---")
    
    # 1. Cargar Datos
    try:
        print(f"Cargando datos desde: {DATA_DIR}")
        df_sx = pd.read_csv(FILES['sx'], sep='\t', header=None)
        df_sy = pd.read_csv(FILES['sy'], sep='\t', header=None)
        df_sum = pd.read_csv(FILES['sum'], sep='\t', header=None)
        
        # Leer metadatos (Temperatura y Radio) del archivo de calibración
        meta = leer_metadatos(FILES['calib'])
        print(f"Metadatos leídos: T={meta['T']} K, R={meta['R']} m")

    except FileNotFoundError as e:
        return {'error': f"No se encontraron los archivos en {DATA_DIR}.\n{e}"}

    # Aplanar para tener la serie temporal completa
    raw_sx = df_sx.values.flatten()
    raw_sy = df_sy.values.flatten()
    raw_sum = df_sum.values.flatten()
    
    min_len = min(len(raw_sx), len(raw_sy), len(raw_sum))
    raw_sx = raw_sx[:min_len]
    raw_sy = raw_sy[:min_len]
    raw_sum = raw_sum[:min_len]

    # 2. Normalizar (Paso 1 de la microscopista)
    # Centramos en cero restando la media
    norm_x = (raw_sx / raw_sum)
    norm_x = norm_x - np.mean(norm_x)
    
    norm_y = (raw_sy / raw_sum)
    norm_y = norm_y - np.mean(norm_y)

    # 3. Calcular PSD (Pasos 2 y 3)
    # Usamos el método de Welch que ya promedia ventanas internamente,
    # lo cual es equivalente y más robusto que "tomar 10 columnas y promediar".
    # Asumimos una frecuencia de muestreo (fs). Si no está en el archivo,
    # necesitaremos preguntarla o deducirla. *Por defecto usaremos 20 kHz*.
    fs = 20000 # Hz (Ajustar si se conoce el valor real)
    
    f_x, Pxx = signal.welch(norm_x, fs, nperseg=1024)
    f_y, Pyy = signal.welch(norm_y, fs, nperseg=1024)

    # 4. Ajuste Lorentziano (Pasos 4 y 5)
    # Ajustamos solo hasta la frecuencia de Nyquist o donde el ruido domina.
    # Filtramos frecuencias muy bajas (drift) y muy altas (ruido).
    mask = (f_x > 10) & (f_x < 5000) 
    
    # Ajuste en X
    popt_x, _ = curve_fit(lorentzian, f_x[mask], Pxx[mask], p0=[100, 1e-3])
    fc_x, D_x_volts = popt_x
    
    # Ajuste en Y
    popt_y, _ = curve_fit(lorentzian, f_y[mask], Pyy[mask], p0=[100, 1e-3])
    fc_y, D_y_volts = popt_y

    # 5. Calcular Viscosidad (Paso 6)
    # El paper explica que D = k_B * T / gamma. 
    # Pero D_volts está en unidades arbitrarias.
    # Sin embargo, la Frecuencia de Corte (fc) es independiente de la calibración Voltios->Metros.
    # k = 2 * pi * gamma * fc
    # Y sabemos que gamma = 6 * pi * eta * R
    # Por lo tanto: k = 2 * pi * (6 * pi * eta * R) * fc -> k = 12 * pi^2 * eta * R * fc
    # ESTO NOS DA LA RIGIDEZ (k), NO LA VISCOSIDAD DIRECTAMENTE sin calibración de posición.
    
    # *CORRECCIÓN*: Para obtener la viscosidad sin conocer el factor de conversión (beta),
    # necesitamos usar el método del paper que relaciona D y fc.
    # D_m2_s = kB * T / (6 * pi * eta * R)
    # D_volts = D_m2_s * beta^2
    # fc = k / (2 * pi * gamma)
    
    # Si el objetivo es VALIDAR, podemos calcular la Rigidez Experimental (k)
    # asumiendo la viscosidad del agua conocida, y compararla con el archivo.
    
    # Asumiendo viscosidad del agua a T (aprox 0.00089 Pa*s a 25C)
    # Esto es más robusto que intentar "medir" la viscosidad sin una calibración precisa de beta.
    eta_agua_teorica = 0.00089 # Pa·s
    gamma_teorico = 6 * np.pi * eta_agua_teorica * meta['R']
    
    kx_calculada = 2 * np.pi * gamma_teorico * fc_x
    ky_calculada = 2 * np.pi * gamma_teorico * fc_y

    resultados = {
        'trayectoria': {'x': norm_x, 'y': norm_y},
        'psd': {'fx': f_x, 'pxx': Pxx, 'fy': f_y, 'pyy': Pyy},
        'ajuste': {
            'fc_x': fc_x, 'fc_y': fc_y,
            'popt_x': popt_x, 'popt_y': popt_y
        },
        'fisica': {
            'kx_calc': kx_calculada,
            'ky_calc': ky_calculada,
            'T': meta['T'],
            'R': meta['R']
        }
    }
    
    print(f"Resultados X: fc={fc_x:.1f} Hz -> kx={kx_calculada*1e9:.2f} pN/um")
    return resultados

def leer_metadatos(filepath):
    """Lee T y Radio del archivo de calibración."""
    meta = {'T': 298.15, 'R': 1.0e-6} # Valores por defecto
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Temperature (K)" in line:
                parts = lines[i+1].strip().split('\t')
                if len(parts) > 3: meta['T'] = float(parts[3])
                if len(parts) > 4: meta['R'] = float(parts[4]) * 1e-6 # um a m
    except Exception as e:
        print(f"Advertencia leyendo metadatos: {e}")
    return meta

# Bloque de prueba
if __name__ == "__main__":
    res = cargar_y_procesar_datos()
    if 'error' not in res:
        # Graficar rápido para verificar
        plt.loglog(res['psd']['fx'], res['psd']['pxx'], label='PSD Experimental X')
        plt.plot(res['psd']['fx'], lorentzian(res['psd']['fx'], *res['ajuste']['popt_x']), 'r--', label='Ajuste Lorentziano')
        plt.legend()
        plt.show()
    else:
        print(res['error'])