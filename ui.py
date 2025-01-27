from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox)
import LLMcalc

class LLMCalculatorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Calculator")
        self.setMinimumSize(600, 400)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Model input section
        model_layout = QHBoxLayout()
        model_label = QLabel("Model ID:")
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("e.g., microsoft/phi-4")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        layout.addLayout(model_layout)
        
        # Hardware override section
        hw_layout = QHBoxLayout()
        
        vram_label = QLabel("VRAM (GB):")
        self.vram_input = QLineEdit()
        self.vram_input.setPlaceholderText("Optional")
        
        bandwidth_label = QLabel("Bandwidth (GB/s):")
        self.bandwidth_input = QLineEdit()
        self.bandwidth_input.setPlaceholderText("Optional")
        
        gpu_count_label = QLabel("GPU Count:")
        self.gpu_count = QComboBox()
        self.gpu_count.addItems([str(i) for i in range(1, 9)])
        
        hw_layout.addWidget(vram_label)
        hw_layout.addWidget(self.vram_input)
        hw_layout.addWidget(bandwidth_label)
        hw_layout.addWidget(self.bandwidth_input)
        hw_layout.addWidget(gpu_count_label)
        hw_layout.addWidget(self.gpu_count)
        layout.addLayout(hw_layout)
        
        # Calculate button
        self.calc_button = QPushButton("Calculate")
        self.calc_button.clicked.connect(self.calculate)
        layout.addWidget(self.calc_button)
        
        # Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)
        
        # System info display
        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(100)
        layout.addWidget(self.system_info)
        
        self.update_system_info()

    def update_system_info(self):
        """Update system information display"""
        total_ram = LLMcalc.get_ram_specs()
        vram, bandwidth = LLMcalc.get_vram_specs()
        ram_bandwidth = LLMcalc.get_memory_bandwidth()
        
        info = f"System Information:\n"
        info += f"Total RAM: {total_ram:.2f} GB\n"
        info += f"VRAM: {vram:.2f} GB\n"
        info += f"GPU Bandwidth: ~{bandwidth} GB/s\n"
        info += f"RAM Bandwidth: {ram_bandwidth:.2f} GB/s"
        
        self.system_info.setText(info)

    def calculate(self):
        """Perform calculation and display results"""
        model_id = self.model_input.text().strip()
        if not model_id:
            self.results_display.setText("Please enter a model ID")
            return
            
        # Get model parameters
        params_text = LLMcalc.get_model_params(model_id)
        if not params_text:
            self.results_display.setText("Could not determine model parameters")
            return
            
        params_b = LLMcalc.convert_params_to_b(params_text)
        
        # Get system specs
        total_ram = LLMcalc.get_ram_specs()
        vram, bandwidth = LLMcalc.get_vram_specs()
        
        # Apply overrides if provided
        try:
            if self.vram_input.text().strip():
                vram = float(self.vram_input.text())
            if self.bandwidth_input.text().strip():
                bandwidth = float(self.bandwidth_input.text())
                
            gpu_count = int(self.gpu_count.currentText())
            if gpu_count > 1:
                vram *= gpu_count
                bandwidth = (bandwidth * gpu_count) * 0.42
        except ValueError:
            self.results_display.setText("Invalid number format in overrides")
            return
            
        ram_bandwidth = LLMcalc.get_memory_bandwidth()
        
        # Calculate results
        results = LLMcalc.analyze_all_quantizations(params_b, vram, bandwidth, total_ram, ram_bandwidth)
        
        # Display results
        output = f"Model Parameters: {params_text} ({params_b / 1e9:.2f}B params)\n\n"
        output += f"Analysis for each quantization level:\n"
        
        for quant, data in results.items():
            output += f"\n{quant.upper()}:\n"
            output += f"Run Type: {data['run_type']}\n"
            output += f"Memory Required: {data['memory_required']:.2f} GB\n"
            if data['offload_percentage'] > 0:
                output += f"GPU Offload Percentage: {100-data['offload_percentage']:.1f}%\n"
            if data['tk/s']:
                output += f"Estimated tk/s: {data['tk/s']:.2f}\n"
                
        self.results_display.setText(output) 
