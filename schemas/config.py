# aqui el codigo relacionado a la inferencia del modelo para mandar a produccion, aqui se manejan los rangos de seguridad y 
# copias de las estructuras basicas de configuracion, el modelo es no dependiente de torch y podria ser exportado a C++ para
# maximo rendimineto, respentando las configuraciones de seguridad

from colorama import Fore, init
init(autoreset=True)
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
import numpy as np


class Log:
    """
    Utilidad estática para impresión rápida de banderas en consola.
    Compatible con tipado estricto (Pyright) al usar f-strings internos.
    """
    
    @staticmethod
    def success(msg: str) -> None:
        print(f"{Fore.LIGHTMAGENTA_EX}[SUCCESS] {msg} ...")
        
    @staticmethod
    def error(msg: str) -> None:
        print(f"{Fore.RED}[ERROR] !! {msg}")
    
    @staticmethod
    def ok(msg:str)->None:
        print(f"{Fore.GREEN} [OK] {msg}")
    @staticmethod
    def info(msg: str) -> None:
        print(f"{Fore.CYAN}[INFO] {msg}")
        
    @staticmethod
    def warn(msg: str) -> None:
        print(f"{Fore.YELLOW}[WARN] ! {msg}")

@dataclass(frozen=True)
class StructureInputConfig:  # Corregido el nombre de la clase
    """
    Clase de configuración para la estructura de entrada del modelo.
    """
    NAME_VARS: Tuple[str, ...] = (
        "tau",
        "DaH",
        "NTU",
        "gamma_r",
        "A",
        "theta_i",
        # "X_i",
        # "X0",
        "theta0"
    )

@dataclass(frozen=True)
class StructureOutputConfig:
    """
    Clase de configuración para la estructura de salida del modelo.
    """
    NAME_VARS: Tuple[str, ...] = (
        "theta",
        # "X"
    )

@dataclass(frozen=True)
class ConfiguracionCondicionesNominales:
    """
    Aqui están las condiciones nominales a un determinado flujo de funcionamiento para la alimentación CSTR
    """
    qf: float = 1.0
    var_qf: float = 0.2

    DaH: float = 0.5
    NTU: float = 1.0
    gamma_r: float = 0.1
    A: float = 1.0

@dataclass(frozen=True)
class ColocationConfig:
    """
    Clase de configuración para la colocalización física del modelo.
    """
    # Aquí puedes agregar parámetros específicos para la colocalización si es necesario
    tau_max: float = 10.0
    theta_i_range: Tuple[float, float] | float= (0.5, 1.5)
    theta_0_range: Tuple[float, float] | float= (0.01, 1.5)
    X_i_range: Dict[str, Tuple|float] = field(default_factory=lambda: {
        "A": (0.1, 0.6),
        "B": (0.1, 0.6),
        "C": (0.1, 0.6)
    })
    
    X_0_range: Dict[str, Tuple|float] = field(default_factory=lambda: {
        "A": (0.1, 0.6),
        "B": (0.1, 0.6),
        "C": (0.1, 0.6)
    })

    nominal_conditions: ConfiguracionCondicionesNominales = field(
        default_factory=ConfiguracionCondicionesNominales
    )

@dataclass(frozen=True)
class ChemicalStructureConfig:
    """
    Clase de configuración para la estructura química del modelo.
    """
    NAME_COMPONENTS: Tuple[str, ...] = ("A", "B", "C")
    ESTEQUIOMETRIC_COEFFS: Dict[str, int] = field(default_factory=lambda: {"A": -1, "B": -1, "C": 2})

    def __post_init__(self):
        nombres_ordenados = tuple(sorted(self.NAME_COMPONENTS))
        coeficientes_ordenados = {k: self.ESTEQUIOMETRIC_COEFFS[k] for k in nombres_ordenados}
        object.__setattr__(self, 'NAME_COMPONENTS', nombres_ordenados)
        object.__setattr__(self, 'ESTEQUIOMETRIC_COEFFS', coeficientes_ordenados)
    
    @property
    def N_COMPONENTS(self) -> int:
        """
        Retorna el número de componentes químicos.
        """
        return len(self.NAME_COMPONENTS)
    
    def __reaction_rate(self, X:Dict[str, Any])->Any:
        """
        Sistema de reacción quimica del reactor
        """
        return X.get("A", 1.0)*X.get("B", 1.0)
    
    def __arrhenius_factor(self, theta: Any, A: Any, gamma: Any) -> Any:
        """
        Factor de Arrhenius adimensional numéricamente estable.
        """
        exp_fn = np.exp
        return A * exp_fn( - gamma / (1.0 + theta))
    
    def energy_balance(
        self, 
        theta: Any, 
        theta_i: Any, 
        X: Dict[str, Any], 
        A: Any, 
        gamma_r: Any, 
        DaH: Any, 
        NTU: Any
    ) -> Any:
        """
        Calcula dtheta/dtau para la temperatura adimensional.
        """
        r_prime = self.__reaction_rate(X)
        k = self.__arrhenius_factor(theta, A, gamma_r)
        react_term = k * r_prime

        conveccion = theta_i - theta
        generacion = DaH * react_term
        enfriamiento = NTU * theta

        return conveccion + generacion - enfriamiento
    
    def mass_balance(
        self, 
        X: Dict[str, Any], 
        X_i: Dict[str, Any], 
        theta: Any, 
        A: Any, 
        gamma_r: Any
    ) -> Dict[str, Any]:
        """
        Calcula dX/dtau para todos los componentes químicos.
        """
        r_prime = self.__reaction_rate(X)
        k = self.__arrhenius_factor(theta, A, gamma_r)
        react_term = k * r_prime

        dX_dict = {}
        for name in self.NAME_COMPONENTS:
            nu_i = self.ESTEQUIOMETRIC_COEFFS.get(name, 0.0)
            # Balance: Entrada - Salida + Generación
            dX_dict[name] = (X_i.get(name, 0.0) - X[name]) + nu_i * react_term

        return dX_dict

@dataclass(frozen=True)
class ConfigManager:
    """
    Clase de configuración para la gestión de parámetros del modelo.
    """
    INPUT_CONFIG: StructureInputConfig = StructureInputConfig()
    OUTPUT_CONFIG: StructureOutputConfig = StructureOutputConfig()
    CHEMICAL_CONFIG: ChemicalStructureConfig = ChemicalStructureConfig()
    COLOCATION_CONFIG: ColocationConfig = ColocationConfig()