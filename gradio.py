import gradio as gr
import pandas as pd
import LLMcalc

# Predefined GPU presets {name: (vram_gb, bandwidth_gbs)}
# Feel free to edit or expand with your own known values:
GPU_PRESETS = {
    "No Preset / Auto-Detect": (0, 0),       # Means "use whatever LLMcalc detects"
    "Nvidia RTX 3060":         (12, 360),    # ~360 GB/s
    "Nvidia RTX 3070":         (8, 448),     # ~448 GB/s
    "Nvidia RTX 3080":         (10, 760),    # ~760 GB/s
    "Nvidia RTX 3090":         (24, 936),    # ~936 GB/s
    "Nvidia RTX 4070":         (12, 504),    # ~504 GB/s
    "Nvidia RTX 4090":         (24, 1008),   # ~1008 GB/s
    "AMD RX 6800":             (16, 512),    # ~512 GB/s
    "AMD RX 6900 XT":          (16, 576),    # ~576 GB/s
    "AMD RX 7900 XTX":         (24, 960),    # ~960 GB/s
    "Apple M2 Max":            (32, 400),    # ~400 GB/s
    "Apple M1 Ultra":          (128, 800),   # ~800 GB/s
    "Intel Arc A770":          (16, 560),    # ~560 GB/s (approx)
}

def analyze_model(model_id, gpu_preset, vram_override, bandwidth_override, gpu_count):
    """
    Perform an LLM usage analysis using LLMcalc, returning system info and
    a DataFrame of quantization results.
    """
    # 1. Fetch the model config & parameters
    params_text = LLMcalc.get_model_params(model_id)
    if not params_text:
        return (f"Could not determine parameters for '{model_id}'. "
                "Check that the model ID is valid on HuggingFace."), pd.DataFrame()

    params_b = LLMcalc.convert_params_to_b(params_text)
    config_data = LLMcalc.fetch_model_config(model_id, params_b / 1e9)

    # 2. Gather system specs
    total_ram = LLMcalc.get_ram_specs()             # System RAM (GB)
    detected_vram, detected_bw, detected_num_gpus, gpu_brand = LLMcalc.get_vram_specs()
    ram_bandwidth = LLMcalc.get_memory_bandwidth()  # System memory bandwidth (GB/s)

    # 3. Check GPU preset override
    if gpu_preset and gpu_preset in GPU_PRESETS:
        preset_vram, preset_bw = GPU_PRESETS[gpu_preset]
        # If user selected a preset other than "No Preset", override with that
        if gpu_preset != "No Preset / Auto-Detect":
            detected_vram = preset_vram
            detected_bw = preset_bw

    # 4. Apply user numeric overrides if provided
    try:
        # Fallback to detected/preset if user leaves it blank or zero
        if vram_override is not None and vram_override > 0:
            detected_vram = vram_override
        if bandwidth_override is not None and bandwidth_override > 0:
            detected_bw = bandwidth_override
        if gpu_count is not None and gpu_count > 0:
            detected_num_gpus = gpu_count
    except:
        return "Invalid override values. Please check your inputs.", pd.DataFrame()

    # If user has multiple GPUs, approximate combined bandwidth
    original_bw = detected_bw
    combined_bw = 0
    coef = 1
    for i in range(detected_num_gpus):
        combined_bw += original_bw * coef
        coef = 0.42
    detected_bw = combined_bw
    # Multiply VRAM by # of GPUs (assuming identical cards):
    total_vram = detected_vram * detected_num_gpus

    # 5. Analyze the model for all quantization levels
    results = LLMcalc.analyze_all_quantizations(
        params_b=params_b,
        vram_gb=total_vram,
        bandwidth=detected_bw,
        ram_gb=total_ram,
        ram_bandwidth=ram_bandwidth,
        config_data=config_data
    )

    # 6. Build a system info string
    system_info = (
        f"**System Info**\n"
        f" - Model ID: `{model_id}`\n"
        f" - Model Params: ~{params_b/1e9:.2f}B\n"
        f" - System RAM: {total_ram:.2f} GB\n"
        f" - System RAM Bandwidth: {ram_bandwidth:.2f} GB/s\n"
        f" - GPU(s): {detected_num_gpus}x {gpu_brand} ({detected_vram:.2f} GB each)\n"
        f" - Est. GPU BW (total): {detected_bw:.2f} GB/s\n"
    )

    # 7. Convert results dict to a DataFrame
    table_data = []
    for quant, data in results.items():
        run_type = data["run_type"]
        mem_req = data["memory_required"]
        offload = data["offload_percentage"]  # how much is going to system RAM
        tks = data["tk/s"]
        ctx_len = data["context"]
        row = {
            "Quant": quant,
            "Run Type": run_type,
            "Mem Required (GB)": f"{mem_req:.2f}",
            "Offload (%)": f"{offload:.1f}" if offload else "0",
            "tk/s": f"{tks:.2f}" if tks else "N/A",
            "Max Context": str(ctx_len) if ctx_len else "N/A",
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    return system_info, df

with gr.Blocks(title="LLM Usage Estimator") as demo:
    gr.Markdown("# LLM Usage Estimator (Gradio Version)")
    gr.Markdown(
        "Enter a Hugging Face model ID (e.g. `microsoft/phi-4`) and optionally "
        "select a GPU Preset or override your GPU VRAM/Bandwidth/Count. "
        "Then click **Analyze**."
    )

    with gr.Row():
        model_id = gr.Textbox(
            label="Hugging Face Model ID",
            value="microsoft/phi-4",
            placeholder="username/repo_name"
        )

    with gr.Row():
        preset_dropdown = gr.Dropdown(
            choices=list(GPU_PRESETS.keys()),
            value="No Preset / Auto-Detect",
            label="GPU Preset"
        )

        vram_input = gr.Number(label="VRAM per GPU (GB)", value=0, precision=1)
        bw_input = gr.Number(label="GPU Bandwidth (GB/s)", value=0, precision=1)
        gpu_count_input = gr.Number(label="GPU Count", value=1, precision=0)

    analyze_btn = gr.Button("Analyze")

    system_info_output = gr.Markdown()
    results_output = gr.DataFrame(
        headers=["Quant", "Run Type", "Mem Required (GB)", "Offload (%)", "tk/s", "Max Context"],
        datatype=["str", "str", "str", "str", "str", "str"],
        label="Quantization Results",
    )

    analyze_btn.click(
        fn=analyze_model,
        inputs=[model_id, preset_dropdown, vram_input, bw_input, gpu_count_input],
        outputs=[system_info_output, results_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
