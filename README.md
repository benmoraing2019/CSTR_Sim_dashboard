
# CSTR Digital Twin: Real-Time Simulation via Physics-Informed Neural Operators (PINO)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.16-005CED?logo=onnx&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit&logoColor=white)
![SciML](https://img.shields.io/badge/Scientific-Machine_Learning-8A2BE2)

Este repositorio contiene la arquitectura de inferencia y la interfaz de usuario (Dashboard) de un **Gemelo Digital** para un Reactor Continuo de Tanque Agitado (CSTR). 

A diferencia de los simuladores tradicionales que dependen de costosos integradores numéricos para resolver Ecuaciones Diferenciales Ordinarias (EDOs) rígidas, este modelo utiliza un **Operador Neuronal Informado por la Física (PINO)** pre-entrenado y exportado a un grafo estático de `ONNX`. El resultado es una simulación física rigurosa ejecutada en **fracciones de milisegundo**.

## Propuesta de Valor y Desempeño
El cuello de botella clásico en la optimización de plantas químicas es el tiempo de cálculo termodinámico. Los controladores predictivos (MPC) necesitan evaluar miles de trayectorias por segundo. 

*   **Integradores Numéricos (Radau/LSODA):** Alta precisión, pero computacionalmente lentos al resolver el término exponencial de Arrhenius iterativamente.
*   **Este Modelo (SciML + ONNX):** Resuelve los balances acoplados a una velocidad promedio de **~30 microsegundos por estado termodinámico**. Permite simular **1,000,000 de escenarios operativos en apenas 30 segundos** en CPU estándar, habilitando la optimización y control predictivo en estricto tiempo real.
*   **Safety Guardrails:** Incluye un sistema de detección de Out-of-Distribution (OOD) nativo en NumPy que alerta si las variables de entrada exceden la envolvente termodinámica de entrenamiento.

---

## Fundamento Matemático (Balances CSTR)

El modelo fue entrenado para respetar estrictamente las leyes de conservación de la materia y energía, modelando la cinética de reacción $A + B \rightarrow C$. Las ecuaciones gobernantes (adimensionalizadas) que el motor neuronal aproxima son:

### 1. Balance de Energía (Evolución Térmica)
Describe la evolución temporal de la temperatura adimensional ($\theta$) considerando la convección, el calor de reacción (generación) y el sistema de refrigeración:

$$
\frac{d\theta}{d\tau} = (\theta_i - \theta) + Da_H \cdot \kappa(\theta) \cdot r'(\mathbf{X}) - NTU \cdot \theta
$$

### 2. Balance de Masa (Especies Químicas)
Describe el consumo y generación de las especies $j \in \{A, B, C\}$ según sus coeficientes estequiométricos ($\nu_j$):

$$
\frac{dX_j}{d\tau} = (X_{j,i} - X_j) + \nu_j \cdot \kappa(\theta) \cdot r'(\mathbf{X})
$$

### 3. Cinética y Factor de Arrhenius
La tasa de reacción $r'(\mathbf{X})$ y la dependencia exponencial de la temperatura fuertemente no-lineal (Arrhenius):

$$
r'(\mathbf{X}) = X_A \cdot X_B
$$

$$
\kappa(\theta) = A \cdot \exp\left( \frac{-\gamma_r}{1 + \theta} \right)
$$

**Donde:**
*   $\tau$: Tiempo adimensional.
*   $Da_H$: Número de Damköhler térmico.
*   $NTU$: Número de Unidades de Transferencia Térmica.
*   $\gamma_r$: Energía de activación adimensional.
*   $\theta_i, X_{j,i}$: Condiciones de alimentación.

---

## Arquitectura de Software

Para garantizar la máxima velocidad en producción y un entorno ligero en la nube, el modelo fue desvinculado de su entorno de entrenamiento (PyTorch):

1.  **Orquestador UI:** Desarrollado en `Streamlit` para interacción web fluida.
2.  **Estructura OOD & Configuración:** Clases tipo `dataclass` manejadas con **NumPy puro**.
3.  **Motor de Inferencia:** `ONNX Runtime` con ejes dinámicos (`dynamic_axes`) habilitados, permitiendo el procesamiento por lotes masivos (Batching) dinámico sin penalización por forma del tensor.
4.  **Visualización:** `Plotly` para exploración interactiva de los perfiles térmicos y másicos.

---

## Instalación y Uso Local

La aplicación está preparada para ser ejecutada nativamente o dentro de un contenedor Docker/DevContainer.

### Opción 1: Ejecución local con Python
```bash
# 1. Clonar el repositorio
git clone [https://github.com/benmoraing2019/CSTR_Sim_dashboard.git](https://github.com/benmoraing2019/CSTR_Sim_dashboard.git)
cd CSTR_Sim_dashboard

# 2. Instalar dependencias estrictas (Solo inferencia)
pip install -r requirements.txt

# 3. Lanzar el Gemelo Digital
streamlit run app.py