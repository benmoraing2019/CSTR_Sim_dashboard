# Model classes for the CSTR inference pipeline

import onnxruntime as ort
from pathlib import Path
from typing import List

from schemas.structure import ONNXInputParams, ONNXOutputParams
from schemas.config import ConfigManager

CONFIG_ONNX = ConfigManager()

class CSTRONNXModel:
    def __init__(self, model_dir: str | Path, name_components: List[str] | None = CONFIG_ONNX.CHEMICAL_CONFIG.NAME_COMPONENTS):
        model_dir = Path(model_dir)
        onnx_files = list(model_dir.glob("*.onnx"))
        
        if not onnx_files:
            raise FileNotFoundError(f"No se encontró ningún archivo .onnx en el directorio: {model_dir}")
            
        self.model_path = str(onnx_files[0])
        
        # onnxruntime detecta y carga el .onnx.data automáticamente si está en el mismo directorio
        self.session = ort.InferenceSession(self.model_path)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.name_components = name_components

    def predict(self, inputs: ONNXInputParams) -> ONNXOutputParams:
        input_array = inputs.to_array()
        
        ort_inputs = {self.input_name: input_array}
        ort_outs = self.session.run([self.output_name], ort_inputs)
        
        output_array = ort_outs[0]
        
        return ONNXOutputParams.from_array(output_array, self.name_components)

    def __call__(self, inputs: ONNXInputParams) -> ONNXOutputParams:
        return self.predict(inputs)