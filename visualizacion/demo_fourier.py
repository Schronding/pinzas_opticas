import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def animar_intuicion_dft():
    
    # señal compuesta 3 Hz y 5 Hz
    fs = 100
    t = np.linspace(0, 2, fs * 2) # 2 segundos
    senal = np.cos(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 5 * t)
    
    freqs_prueba = np.linspace(0, 8, 400) # Probamos de 0 a 8 Hz

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Intuición de la DFT: El Centro de Masa", fontsize=16, color="white")
    gs = GridSpec(1, 2, figure=fig)

    ax_polar = fig.add_subplot(gs[0, 0])
    ax_polar.set_title("Señal Enrollada en el Plano Complejo")
    ax_polar.set_xlim(-2, 2); ax_polar.set_ylim(-2, 2)
    ax_polar.grid(True, alpha=0.3)
    ax_polar.set_aspect('equal')
    
    line_wrap, = ax_polar.plot([], [], color='cyan', lw=1, alpha=0.8)
    dot_mass, = ax_polar.plot([], [], 'ro', markersize=8, label='Centro de Masa')
    ax_polar.legend(loc='upper right')

    ax_freq = fig.add_subplot(gs[0, 1])
    ax_freq.set_title("Magnitud de la Transformada (Posición del Centro de Masa)")
    ax_freq.set_xlim(0, 8)
    ax_freq.set_ylim(0, 1)
    ax_freq.set_xlabel("Frecuencia de Prueba (Hz)")
    ax_freq.grid(True, alpha=0.3)
    
    line_dft, = ax_freq.plot([], [], color='yellow', lw=2)
    vline_current, = ax_freq.plot([], [], 'r--', alpha=0.5)
    
    fig.patch.set_facecolor('#101010')
    for ax in [ax_polar, ax_freq]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('white')

    magnitudes = []
    
    def update(frame):
        f_probe = freqs_prueba[frame]
        
        # ECUACIÓN CLAVE DE LA DFT: e^(-i * 2pi * f * t)
        # Enrollamos la señal alrededor del origen a la frecuencia f_probe
        vector_complejo = senal * np.exp(-1j * 2 * np.pi * f_probe * t)
        
        # Coordenadas para graficar
        x_vals = np.real(vector_complejo)
        y_vals = np.imag(vector_complejo)
        
        # El valor de la Transformada es el PROMEDIO (Centro de Masa)
        center_x = np.mean(x_vals)
        center_y = np.mean(y_vals)
        magnitude = np.sqrt(center_x**2 + center_y**2)
        
        if len(magnitudes) <= frame:
            magnitudes.append(magnitude)
            
        # Dibujar
        line_wrap.set_data(x_vals, y_vals)
        dot_mass.set_data([center_x], [center_y])
        
        line_dft.set_data(freqs_prueba[:len(magnitudes)], magnitudes)
        vline_current.set_data([f_probe, f_probe], [0, 1])
        
        return line_wrap, dot_mass, line_dft, vline_current

    ani = animation.FuncAnimation(fig, update, frames=len(freqs_prueba), interval=20, blit=True)
    plt.show()

def animar_carrera_fft():
    
    N_vals = np.arange(1, 100, 1)
    
    # Complejidad Teórica
    ops_dft = N_vals**2             # O(N^2)
    ops_fft = N_vals * np.log2(N_vals) # O(N log N)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("La Diferencia Real: DFT vs FFT (Número de Operaciones)", fontsize=14, color='white')
    
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#101010')
    ax.grid(True, color='gray', alpha=0.2)
    
    ax.set_xlabel("Número de Datos (N)", color='white')
    ax.set_ylabel("Operaciones Requeridas", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('white')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10000) 
    
    line_dft, = ax.plot([], [], 'r-', lw=3, label='DFT Clásica ($N^2$)')
    line_fft, = ax.plot([], [], 'g-', lw=3, label='FFT ($N \log N$)')
    
    point_dft, = ax.plot([], [], 'ro', markersize=10)
    point_fft, = ax.plot([], [], 'go', markersize=10)
    
    txt_stats = ax.text(0.05, 0.8, '', transform=ax.transAxes, color='white', fontsize=12,
                        bbox=dict(facecolor='#333', alpha=0.8))
    
    ax.legend(facecolor='#333', labelcolor='white')

    def update(frame):
        current_N = N_vals[:frame]
        
        dft_y = ops_dft[:frame]
        fft_y = ops_fft[:frame]
        
        line_dft.set_data(current_N, dft_y)
        line_fft.set_data(current_N, fft_y)
        
        if frame > 0:
            idx = frame - 1
            point_dft.set_data([N_vals[idx]], [ops_dft[idx]])
            point_fft.set_data([N_vals[idx]], [ops_fft[idx]])
            
            ratio = ops_dft[idx] / (ops_fft[idx] + 0.001)
            txt_stats.set_text(
                f"N = {N_vals[idx]}\n"
                f"Operaciones DFT: {int(ops_dft[idx])}\n"
                f"Operaciones FFT: {int(ops_fft[idx])}\n"
                f"FFT es {ratio:.1f}x más rápida aquí"
            )
            
        return line_dft, line_fft, point_dft, point_fft, txt_stats

    ani = animation.FuncAnimation(fig, update, frames=len(N_vals), interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    animar_intuicion_dft()
    animar_carrera_fft()