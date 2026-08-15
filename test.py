from .schemas.structure import ONNXInputParams, ONNXOutputParams
from .schemas.model import CSTRONNXModel
from .schemas.config import Log, ConfigManager
import os

def prueba_carga_modelo():
    # Ruta al directorio que contiene el archivo .onnx
    model_dir = "dist/model"
    
    # Crear una instancia del modelo
    modelo = CSTRONNXModel(model_dir)
    
    # Verificar que la ruta del modelo se haya cargado correctamente
    assert modelo.model_path.endswith(".onnx"), "El archivo del modelo no es un archivo .onnx"
    
    print("Modelo cargado correctamente desde:", modelo.model_path)

    return modelo

def prueba_prediccion(modelo: CSTRONNXModel):
    # Crear un conjunto de datos de prueba con valores aleatorios
    import numpy as np
    
    # Definir los nombres de los componentes químicos
    name_components = modelo.name_components
    
    # Crear datos de entrada aleatorios
    tau = np.random.rand(1, 1).astype(np.float32)
    DaH = np.random.rand(1, 1).astype(np.float32)
    NTU = np.random.rand(1, 1).astype(np.float32)
    gamma_r = np.random.rand(1, 1).astype(np.float32)
    A = np.random.rand(1, 1).astype(np.float32)
    theta_i = np.random.rand(1, 1).astype(np.float32)
    theta0 = np.random.rand(1, 1).astype(np.float32)
    
    # Crear diccionarios para X_i y X0 con valores aleatorios
    X_i = {name: np.random.rand(1, 1).astype(np.float32) for name in name_components}
    X0 = {name: np.random.rand(1, 1).astype(np.float32) for name in name_components}
    
    # Crear una instancia de ONNXInputParams con los datos de prueba
    input_params = ONNXInputParams(
        tau=tau,
        DaH=DaH,
        NTU=NTU,
        gamma_r=gamma_r,
        A=A,
        theta_i=theta_i,
        theta0=theta0,
        X_i=X_i,
        X0=X0
    )
    
    # Realizar la predicción utilizando el modelo cargado
    output_params = modelo.predict(input_params)
    
    # Verificar que la salida tenga la forma esperada
    assert output_params.to_array().shape[0] == 1, "La salida no tiene la forma esperada"
    
    Log.ok(f"Predicción realizada correctamente. Salida: {output_params.to_array()}")

def prueba_estres_batch_model(modelo: CSTRONNXModel, batch_size: int = 1000):
    import numpy as np
    """"Evaluar tiempos de carga y predicción del modelo con un batch grande de datos de entrada."""
    from time import time

    tiempo_inicio = time()
    # Definir los nombres de los componentes químicos
    name_components = modelo.name_components
    
    # Crear datos de entrada aleatorios para un batch grande
    tau = np.random.rand(batch_size, 1).astype(np.float32)
    DaH = np.random.rand(batch_size, 1).astype(np.float32)
    NTU = np.random.rand(batch_size, 1).astype(np.float32)
    gamma_r = np.random.rand(batch_size, 1).astype(np.float32)
    A = np.random.rand(batch_size, 1).astype(np.float32)
    theta_i = np.random.rand(batch_size, 1).astype(np.float32)
    theta0 = np.random.rand(batch_size, 1).astype(np.float32)
    
    # Crear diccionarios para X_i y X0 con valores aleatorios
    X_i = {name: np.random.rand(batch_size, 1).astype(np.float32) for name in name_components}
    X0 = {name: np.random.rand(batch_size, 1).astype(np.float32) for name in name_components}
    
    # Crear una instancia de ONNXInputParams con los datos de prueba
    input_params = ONNXInputParams(
        tau=tau,
        DaH=DaH,
        NTU=NTU,
        gamma_r=gamma_r,
        A=A,
        theta_i=theta_i,
        theta0=theta0,
        X_i=X_i,
        X0=X0
    )
    
    # Realizar la predicción utilizando el modelo cargado
    output_params = modelo.predict(input_params)
    
    # Verificar que la salida tenga la forma esperada
    assert output_params.to_array().shape[0] == batch_size, "La salida no tiene la forma esperada"

    tiempo_fin = time()
    tiempo_total = tiempo_fin - tiempo_inicio
    Log.info(f"Tiempo total para procesar un batch de tamaño {batch_size}: {tiempo_total:.4f} segundos")
    Log.info(f"Tiempo promedio por muestra: {(tiempo_total / batch_size):.6f} segundos")
    
    Log.ok(f"Prueba de estrés con batch size {batch_size} realizada correctamente. Salida: {output_params.to_array()}")

def prueba_graficos_curvas(modelo: CSTRONNXModel, n_points: int = 100):
    """
    Genera 4 escenarios operativos dentro del rango seguro, evalúa la evolución
    temporal (tau de 0.01 a 10.0) usando ONNX y grafica los resultados con Seaborn.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from time import time


    CONFIG = ConfigManager()
    coloc_cfg = CONFIG.COLOCATION_CONFIG
    
    # 1. Configuración de estilo visual
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colores_escenarios = sns.color_palette("husl", 4)
    
    # Rango temporal de evaluación
    tau_vals = np.linspace(0.01, 10.0, n_points).astype(np.float32)

    for i in range(4):
        # 2. Generación del Escenario Aleatorio (DENTRO DEL RANGO SEGURO)
        DaH_val = np.random.uniform(0.4, 0.6) # Aproximación alrededor del nominal 0.5
        NTU_val = np.random.uniform(0.8, 1.2) # Aproximación alrededor del nominal 1.0
        gamma_r_val = 0.1 # Constante
        A_val = 1.0       # Constante
        
        # Muestreo de temperaturas iniciales/entrada dentro del rango seguro
        theta_i_val = np.random.uniform(*coloc_cfg.theta_i_range)
        theta0_val = np.random.uniform(*coloc_cfg.theta_0_range)
        
        # Muestreo de concentraciones dentro del rango seguro
        X_i_vals = {name: np.random.uniform(*rango) for name, rango in coloc_cfg.X_i_range.items()}
        X0_vals = {name: np.random.uniform(*rango) for name, rango in coloc_cfg.X_0_range.items()}

        # 3. Expansión a tensores (n_points, 1) para inferencia vectorizada
        tau = tau_vals.reshape(-1, 1)
        DaH = np.full((n_points, 1), DaH_val, dtype=np.float32)
        NTU = np.full((n_points, 1), NTU_val, dtype=np.float32)
        gamma_r = np.full((n_points, 1), gamma_r_val, dtype=np.float32)
        A = np.full((n_points, 1), A_val, dtype=np.float32)
        theta_i = np.full((n_points, 1), theta_i_val, dtype=np.float32)
        theta0 = np.full((n_points, 1), theta0_val, dtype=np.float32)
        
        X_i = {name: np.full((n_points, 1), val, dtype=np.float32) for name, val in X_i_vals.items()}
        X0 = {name: np.full((n_points, 1), val, dtype=np.float32) for name, val in X0_vals.items()}

        input_params = ONNXInputParams(
            tau=tau, DaH=DaH, NTU=NTU, gamma_r=gamma_r, A=A,
            theta_i=theta_i, theta0=theta0, X_i=X_i, X0=X0
        )

        # 4. Inferencia y medición de tiempo
        t_start = time()
        output_params = modelo.predict(input_params)
        t_end = time()
        
        # Calcular tiempo en nano (µs)
        tiempo_us = (t_end - t_start) * 1_000
        
        # 5. Extracción de resultados para graficar
        theta_out = output_params.theta.flatten()
        X_out = {name: val.flatten() for name, val in output_params.X.items()}
        
        # 6. Graficado
        color = colores_escenarios[i]
        label_base = f"Escenario {i+1} ({tiempo_us:.1f} ms)"
        
        # Gráfico Izquierdo: Balance de Masa (Solo graficamos el componente 'A' para claridad visual)
        axes[0].plot(tau_vals, X_out['A'], color=color, linewidth=2, label=label_base)
        
        # Gráfico Derecho: Balance de Energía
        axes[1].plot(tau_vals, theta_out, color=color, linewidth=2, label=label_base)

    # 7. Detalles estéticos de los gráficos
    axes[0].set_title("Balance de Masa (Conversión de Especie A)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(r"Tiempo Adimensional ($\tau$)", fontsize=12)
    axes[0].set_ylabel(r"Concentración Adimensional ($X_A$)", fontsize=12)
    axes[0].legend(title="Inferencia ONNX (100 pts)", loc="best")
    
    axes[1].set_title("Balance de Energía (Evolución Térmica)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(r"Tiempo Adimensional ($\tau$)", fontsize=12)
    axes[1].set_ylabel(r"Temperatura Adimensional ($\theta$)", fontsize=12)
    axes[1].legend(title="Inferencia ONNX (100 pts)", loc="best")
    
    plt.tight_layout()
    plt.savefig("dist/prueba_curvas_onnx.png", dpi=300)
    plt.close()
    Log.ok("Gráficas de prueba generadas en 'dist/prueba_curvas_onnx.png'")

if __name__ == "__main__":
    Log.success("Iniciando prueba de carga del modelo ONNX...")
    modelo = prueba_carga_modelo()
    Log.ok("Prueba de carga del modelo ONNX completada exitosamente.")

    Log.success("Iniciando prueba de predicción del modelo ONNX...")

    prueba_prediccion(modelo)
    Log.ok("Prueba de predicción del modelo ONNX completada exitosamente.")

    Log.success("Iniciando prueba de estrés con batch grande del modelo ONNX...")
    prueba_estres_batch_model(modelo, batch_size=100000)
    Log.ok("Prueba de estrés con batch grande del modelo ONNX completada exitosamente.")

    Log.success("Iniciando prueba de generación de curvas con el modelo ONNX...")
    prueba_graficos_curvas(modelo, n_points=100)
    Log.ok("Prueba de generación de curvas con el modelo ONNX completada exitosamente.")