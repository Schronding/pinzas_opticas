# -*- coding: utf-8 -*-
"""
Auditor de Integridad de Datos Experimentales.
Realiza pruebas forenses para detectar duplicados, correlaciones sospechosas
y problemas de formato en los archivos de datos.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
# Ajusta la ruta si es necesario para apuntar a tu carpeta 'datos_experimental'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'datos_experimentales')

FILES = {
    'sx': 'datos_sx.dat',
    'sy': 'datos_sy.dat',
    'sum': 'datos_sum.dat'
}

def get_file_hash(filepath):
    """Calcula el hash SHA-256 de un archivo para detectar copias exactas."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Leer en bloques de 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def auditar_datos():
    print(f"🔍 INICIANDO AUDITORÍA EN: {DATA_DIR}\n")
    
    rutas = {k: os.path.join(DATA_DIR, v) for k, v in FILES.items()}
    
    # 1. VERIFICACIÓN DE EXISTENCIA
    missing = [f for f, p in rutas.items() if not os.path.exists(p)]
    if missing:
        print(f"❌ ERROR CRÍTICO: Faltan archivos: {missing}")
        return
    else:
        print("✅ Todos los archivos existen.")

    # 2. VERIFICACIÓN BINARIA (HASHING)
    print("\n--- 2. Análisis de Identidad de Archivos (Hash) ---")
    hashes = {}
    for k, path in rutas.items():
        h = get_file_hash(path)
        hashes[k] = h
        print(f"  📄 {k.upper()}: {h[:16]}...") # Muestra solo el inicio del hash

    if hashes['sx'] == hashes['sy']:
        print("⚠️ ALERTA ROJA: 'datos_sx.dat' y 'datos_sy.dat' son BINARIAMENTE IDÉNTICOS.")
        print("   Esto confirma que copiaste el mismo archivo con dos nombres distintos.")
    else:
        print("✅ Los archivos son binariamente diferentes (buena señal).")

    # 3. ANÁLISIS DE CONTENIDO NUMÉRICO
    print("\n--- 3. Análisis Estadístico de las Señales ---")
    try:
        # Cargar datos
        df_sx = pd.read_csv(rutas['sx'], sep='\t', header=None)
        df_sy = pd.read_csv(rutas['sy'], sep='\t', header=None)
        
        raw_x = df_sx.values.flatten()
        raw_y = df_sy.values.flatten()
        
        # Recortar al mínimo común
        min_len = min(len(raw_x), len(raw_y))
        x = raw_x[:min_len]
        y = raw_y[:min_len]
        
        print(f"  Puntos analizados: {min_len}")
        
        # Calcular Correlación de Pearson
        corr = np.corrcoef(x, y)[0, 1]
        print(f"  Coeficiente de Correlación (Pearson): {corr:.6f}")
        
        if abs(corr) > 0.95:
            print("⚠️ ALERTA: Correlación extremadamente alta (>0.95).")
            print("   Aunque los archivos no sean idénticos byte a byte,")
            print("   contienen prácticamente la misma información física.")
        elif abs(corr) < 0.1:
            print("✅ Correlación baja. Esto es normal para X vs Y en movimiento Browniano.")
        else:
            print(f"ℹ️ Correlación moderada. Revisar si hay astigmatismo fuerte o rotación.")

        # 4. PRUEBA DE DIFERENCIA PUNTO A PUNTO
        diff = np.abs(x - y)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        print(f"\n  Diferencia promedio absoluta |X - Y|: {mean_diff:.6e}")
        print(f"  Diferencia máxima absoluta: {max_diff:.6e}")
        
        if mean_diff < 1e-9:
             print("⚠️ ALERTA: La diferencia numérica es despreciable (cero o error de máquina).")
             print("   Son matemáticamente la misma señal.")

        # 5. VISUALIZACIÓN RÁPIDA (DIAGONAL CHECK)
        print("\n--- Generando gráfico de diagnóstico... ---")
        plt.figure(figsize=(10, 5))
        
        # Scatter plot (X vs Y)
        plt.subplot(1, 2, 1)
        plt.plot(x[:1000], y[:1000], '.', alpha=0.5)
        plt.title("Mapa X-Y (Primeros 1000 ptos)")
        plt.xlabel("Señal X")
        plt.ylabel("Señal Y")
        plt.grid(True)
        
        # Series de tiempo superpuestas
        plt.subplot(1, 2, 2)
        plt.plot(x[:200], label='X')
        plt.plot(y[:200], label='Y', alpha=0.7)
        plt.title("Series de Tiempo (Zoom)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error leyendo datos numéricos: {e}")

if __name__ == "__main__":
    auditar_datos()