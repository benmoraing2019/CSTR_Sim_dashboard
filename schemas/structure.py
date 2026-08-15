import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from .config import ConfigManager

@dataclass(frozen=True)
class ONNXInferenceConfig:
    INPUT_VARS: Tuple[str, ...] = (
        "tau", "DaH", "NTU", "gamma_r", "A", "theta_i", "theta0"
    )
    OUTPUT_VARS: Tuple[str, ...] = ("theta",)
    DEFAULT_COMPONENTS: Tuple[str, ...] = ("A", "B", "C")

CONFIG_ONNX = ConfigManager()

def ensure_2d(arr: np.ndarray | float | int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from .config import ConfigManager, Log

CONFIG = ConfigManager()

def ensure_2d(arr: np.ndarray | float | int) -> np.ndarray:
    """Asegura forma (N, 1) y float32 para ONNX."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

@dataclass
class ONNXInputParams:
    tau: np.ndarray
    DaH: np.ndarray
    NTU: np.ndarray
    gamma_r: np.ndarray
    A: np.ndarray
    theta_i: np.ndarray
    theta0: np.ndarray
    X_i: Dict[str, np.ndarray]
    X0: Dict[str, np.ndarray]

    name_components: List[str] = field(init=False)

    def __post_init__(self):
        self.name_components = sorted(self.X0.keys())
        # Llamamos al sistema de validación de seguridad justo al instanciar
        self._validate_safety_ranges()

    def _check_bounds(self, var_name: str, values: np.ndarray, bounds: Tuple[float, float] | float):
        """
        Función auxiliar que verifica si los valores exceden los límites de entrenamiento.
        """
        if isinstance(bounds, tuple):
            min_val, max_val = bounds
            
            # Buscamos si algún elemento del batch viola los límites
            if np.any(values < min_val) or np.any(values > max_val):
                # Extraemos los valores exactos que fallaron para un reporte detallado
                min_found = np.min(values)
                max_found = np.max(values)
                Log.warn(
                    f"Out-of-Distribution Detectado en '{var_name}': "
                    f"Valores de entrada [{min_found:.4f}, {max_found:.4f}] "
                    f"exceden el rango de entrenamiento seguro [{min_val}, {max_val}]."
                )
        else:
            # Si el límite es un float fijo, verificamos tolerancia (ej. constantes físicas)
            if not np.allclose(values, bounds, atol=1e-4):
                Log.warn(
                    f"Desviación de constante detectada en '{var_name}': "
                    f"Se esperaba exactamente {bounds}, pero se recibió otro valor."
                )

    def _validate_safety_ranges(self):
        """
        Cruza los tensores de entrada contra los rangos de la configuración 
        de colocalización para alertar sobre extrapolaciones peligrosas.
        """
        coloc_cfg = CONFIG.COLOCATION_CONFIG
        
        # 1. Validación de variables independientes (Tiempo y Temperaturas)
        # Nota: tau mínimo asumimos 0.0
        self._check_bounds("tau", self.tau, (0.0, coloc_cfg.tau_max))
        self._check_bounds("theta_i", self.theta_i, coloc_cfg.theta_i_range)
        self._check_bounds("theta0", self.theta0, coloc_cfg.theta_0_range)
        
        # 2. Validación de Concentraciones de Entrada (X_i)
        for name, values in self.X_i.items():
            if name in coloc_cfg.X_i_range:
                self._check_bounds(f"X_i_{name}", values, coloc_cfg.X_i_range[name])
                
        # 3. Validación de Concentraciones Iniciales (X0)
        for name, values in self.X0.items():
            if name in coloc_cfg.X_0_range:
                self._check_bounds(f"X0_{name}", values, coloc_cfg.X_0_range[name])

    def to_array(self) -> np.ndarray:
        scalars_list = [ensure_2d(getattr(self, attr)) for attr in CONFIG.INPUT_CONFIG.NAME_VARS]
        components_i_list = [ensure_2d(self.X_i[name]) for name in self.name_components]
        components_0_list = [ensure_2d(self.X0[name]) for name in self.name_components]
        
        return np.concatenate(scalars_list + components_i_list + components_0_list, axis=1)

    @classmethod
    def from_array(cls, array: np.ndarray, name_components: List[str] | None = None) -> "ONNXInputParams":
        name_components = sorted(name_components or list(CONFIG.CHEMICAL_CONFIG.NAME_COMPONENTS))
        array = ensure_2d(array)
        
        scalars = {}
        X_i = {}
        X0 = {}
        idx = 0
        
        for attr in CONFIG.INPUT_CONFIG.NAME_VARS:
            scalars[attr] = array[:, idx:idx+1]
            idx += 1
            
        for name in name_components:
            X_i[name] = array[:, idx:idx+1]
            idx += 1
            
        for name in name_components:
            X0[name] = array[:, idx:idx+1]
            idx += 1
            
        return cls(**scalars, X_i=X_i, X0=X0)

@dataclass
class ONNXOutputParams:
    theta: np.ndarray
    X: Dict[str, np.ndarray]
    
    name_components: List[str] = field(init=False)

    def __post_init__(self):
        self.name_components = sorted(self.X.keys())

    def to_array(self) -> np.ndarray:
        scalars_list = [ensure_2d(getattr(self, attr)) for attr in CONFIG_ONNX.OUTPUT_CONFIG.NAME_VARS]
        components_list = [ensure_2d(self.X[name]) for name in self.name_components]

        return np.concatenate(scalars_list + components_list, axis=1)

    @classmethod
    def from_array(cls, array: np.ndarray, name_components: List[str] | None = None) -> "ONNXOutputParams":
        name_components = sorted(name_components or list(CONFIG_ONNX.CHEMICAL_CONFIG.NAME_COMPONENTS))
        array = ensure_2d(array)
        
        scalars = {}
        X = {}
        idx = 0
        
        for attr in CONFIG_ONNX.OUTPUT_CONFIG.NAME_VARS:
            scalars[attr] = array[:, idx:idx+1]
            idx += 1
            
        for name in name_components:
            X[name] = array[:, idx:idx+1]
            idx += 1
        return cls(**scalars, X=X)