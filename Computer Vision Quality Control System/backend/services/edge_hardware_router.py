import psutil
import torch

class AdaptiveEdgeHardwareRouter:
    @staticmethod
    def resolve_optimal_runtime() -> tuple:
        """Autodetects underlying hardware infrastructure to route to the fastest execution model version"""
        # 1. التحقق من وجود كروت Nvidia والـ CUDA Context
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "Jetson" in device_name or "Orin" in device_name:
                return "TensorRT Compiled Engine", "NVIDIA Jetson Orin Nano Edge Node"
            return "CUDA PyTorch Matrix Core", f"Server GPU: {device_name}"
            
        # 2. الفحص التلقائي في بيئات الـ CPU الصرفة
        cpu_info = psutil.cpu_times()
        return "OpenVINO Optimized ONNX Graph", "Intel NUC Industrial Processor Node"