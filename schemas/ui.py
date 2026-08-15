import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class SidebarUI:
    theta_i_range: Tuple[float, float]
    theta_0_range: Tuple[float, float]
    X_i_range: Dict[str, Tuple[float, float]]
    X_0_range: Dict[str, Tuple[float, float]]

    def render(self) -> Dict[str, float | int]:
        st.sidebar.header("🎛️ Parámetros Operativos")
        
        n_points = st.sidebar.slider("Puntos de Simulación", 50, 1000, 200, step=50)

        st.sidebar.subheader("🌡️ Temperaturas Adimensionales")
        theta_i = st.sidebar.slider("Temp. Alimentación (θ_i)", 
                                    float(self.theta_i_range[0]), float(self.theta_i_range[1]), 1.0)
        theta0 = st.sidebar.slider("Temp. Inicial Reactor (θ_0)", 
                                   float(self.theta_0_range[0]), float(self.theta_0_range[1]), 0.5)

        st.sidebar.subheader("🧪 Concentraciones de Alimentación")
        Xi_A = st.sidebar.slider("Entrada Especie A", float(self.X_i_range["A"][0]), float(self.X_i_range["A"][1]), 0.5)
        Xi_B = st.sidebar.slider("Entrada Especie B", float(self.X_i_range["B"][0]), float(self.X_i_range["B"][1]), 0.5)

        st.sidebar.subheader("🔬 Concentraciones Iniciales")
        X0_A = st.sidebar.slider("Inicial Especie A", float(self.X_0_range["A"][0]), float(self.X_0_range["A"][1]), 0.2)
        X0_B = st.sidebar.slider("Inicial Especie B", float(self.X_0_range["B"][0]), float(self.X_0_range["B"][1]), 0.2)
        X0_C = st.sidebar.slider("Inicial Especie C", float(self.X_0_range["C"][0]), float(self.X_0_range["C"][1]), 0.1)

        ejecutar = st.sidebar.button("🚀 Calcular Simulación", type="primary", use_container_width=True)
        
        st.sidebar.divider()
        st.sidebar.markdown("**⚙️ Servicios de Ingeniería AI**\nDesarrollo de simuladores ultrarrápidos y gemelos digitales.")

        return {
            "n_points": n_points,
            "theta_i": theta_i, "theta0": theta0,
            "Xi_A": Xi_A, "Xi_B": Xi_B,
            "X0_A": X0_A, "X0_B": X0_B, "X0_C": X0_C,
            "ejecutar": ejecutar
        }

@dataclass
class MetricsUI:
    n_points: int
    tiempo_ms: float
    tiempo_us_por_punto: float

    def render(self):
        st.subheader("⏱️ Rendimiento Computacional")
        col1, col2, col3 = st.columns(3)
        col1.metric("Puntos Simulados", f"{self.n_points}")
        col2.metric("Tiempo Total de Inferencia", f"{self.tiempo_ms:.3f} ms")
        col3.metric("Velocidad por Estado", f"{self.tiempo_us_por_punto:.2f} µs")
        st.divider()

@dataclass
class PlotlyChartsUI:
    tau_vals: list | tuple | object
    X_out: Dict[str, object]
    theta_out: object
    theta_i_val: float

    def render(self):
        col_graph1, col_graph2 = st.columns(2)

        with col_graph1:
            fig_mass = go.Figure()
            fig_mass.add_trace(go.Scatter(x=self.tau_vals, y=self.X_out['A'], mode='lines', name='Especie A', line=dict(color='#e74c3c', width=3)))
            fig_mass.add_trace(go.Scatter(x=self.tau_vals, y=self.X_out['B'], mode='lines', name='Especie B', line=dict(color='#3498db', width=3)))
            fig_mass.add_trace(go.Scatter(x=self.tau_vals, y=self.X_out['C'], mode='lines', name='Especie C', line=dict(color='#2ecc71', width=3)))
            
            fig_mass.update_layout(
                title="Dinámica de Concentraciones",
                xaxis_title="Tiempo Adimensional (τ)", yaxis_title="Concentración",
                hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_mass, use_container_width=True)

        with col_graph2:
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(x=self.tau_vals, y=self.theta_out, mode='lines', name='Temp. Reactor', line=dict(color='#9b59b6', width=3)))
            fig_energy.add_hline(y=self.theta_i_val, line_dash="dash", line_color="gray", annotation_text="Temp. Alimentación")
            
            fig_energy.update_layout(
                title="Dinámica Térmica",
                xaxis_title="Tiempo Adimensional (τ)", yaxis_title="Temperatura (θ)",
                hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_energy, use_container_width=True)