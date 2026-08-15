import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import numpy as np
import time

from schemas.model import CSTRONNXModel
from schemas.structure import ONNXInputParams
from schemas.config import ConfigManager
from schemas.ui import SidebarUI, MetricsUI, PlotlyChartsUI

st.set_page_config(page_title="CSTR Digital Twin | Real-Time SciML", page_icon="⚛️", layout="wide")

@st.cache_resource
def load_resources():
    model_dir = Path("dist/model") if Path("dist/model").exists() else Path("model")
    return CSTRONNXModel(model_dir), ConfigManager()

modelo, config = load_resources()
coloc_cfg = config.COLOCATION_CONFIG

st.title("⚡ Gemelo Digital CSTR: Physics-Informed Neural Operator (PINO)")
st.markdown("Dashboard interactivo de inferencia en tiempo real exportado a **ONNX y NumPy**. Resuelve EDOs rígidas en fracciones de milisegundo.")

sidebar = SidebarUI(
    theta_i_range=coloc_cfg.theta_i_range,
    theta_0_range=coloc_cfg.theta_0_range,
    X_i_range=coloc_cfg.X_i_range,
    X_0_range=coloc_cfg.X_0_range
)
params = sidebar.render()

if params["ejecutar"]:
    n_points = params["n_points"]
    tau_vals = np.linspace(0.01, 10.0, n_points).astype(np.float32)

    tau = tau_vals.reshape(-1, 1)
    DaH = np.full((n_points, 1), 0.5, dtype=np.float32)
    NTU = np.full((n_points, 1), 1.0, dtype=np.float32)
    gamma_r = np.full((n_points, 1), 0.1, dtype=np.float32)
    A = np.full((n_points, 1), 1.0, dtype=np.float32)
    
    theta_i = np.full((n_points, 1), params["theta_i"], dtype=np.float32)
    theta0 = np.full((n_points, 1), params["theta0"], dtype=np.float32)

    X_i = {
        "A": np.full((n_points, 1), params["Xi_A"], dtype=np.float32),
        "B": np.full((n_points, 1), params["Xi_B"], dtype=np.float32),
        "C": np.full((n_points, 1), 0.0, dtype=np.float32)
    }
    X0 = {
        "A": np.full((n_points, 1), params["X0_A"], dtype=np.float32),
        "B": np.full((n_points, 1), params["X0_B"], dtype=np.float32),
        "C": np.full((n_points, 1), params["X0_C"], dtype=np.float32)
    }

    input_params = ONNXInputParams(
        tau=tau, DaH=DaH, NTU=NTU, gamma_r=gamma_r, A=A,
        theta_i=theta_i, theta0=theta0, X_i=X_i, X0=X0
    )

    t_start = time.perf_counter()
    output_params = modelo.predict(input_params)
    t_end = time.perf_counter()

    tiempo_ms = (t_end - t_start) * 1000
    tiempo_us_por_punto = (tiempo_ms * 1000) / n_points

    metrics = MetricsUI(n_points, tiempo_ms, tiempo_us_por_punto)
    metrics.render()

    theta_out = output_params.theta.flatten()
    X_out = {name: val.flatten() for name, val in output_params.X.items()}

    charts = PlotlyChartsUI(tau_vals, X_out, theta_out, params["theta_i"])
    charts.render()

else:
    st.info("👈 Ajusta los parámetros en la barra lateral y presiona **'Calcular Simulación'** para visualizar.")