#LLM Calculator

This script estimates the memory requirements and performance of Hugging Face models based on quantization levels. It fetches model parameters, calculates required memory, and analyzes performance with different RAM/VRAM configurations.

It supports windows and Linux, AMD, Intel, and Nvidia GPUs. You will need smi (cuda toolkit (?)) installed to recognise your Nvidia GPU.

Warning: The tool isn't tested outside of Linux+Nvidia, so results may be inaccurate. It's a rough estimate.
It may or may not work with MultiGPU setups. If not, use -n to specify how many cards you have (assuming they're all the same). If you have an RTX 3070 and an RTX 3060 for example, you may need to specify -v and -b to be the average values of the two. 

**Flags**
```
-b, --bandwidth: Override memory bandwidth (GB/s).
-n, --num-gpus: Number of GPUs (default is 1).
-v, --vram: Override VRAM amount per card in GB.
```
Example:
`python main.py -b 950 -n 2 -v 24`

## How It Works
Enter a Hugging Face model ID (e.g., microsoft/phi-4) to get its parameter count.
The script fetches system RAM and VRAM specs. You can override them with flags.
It analyzes memory requirements for several quantization schemes and estimates throughput (tk/s).
Based on this, it recommends whether the model should run in VRAM, system RAM, or if partial offloading is needed.
