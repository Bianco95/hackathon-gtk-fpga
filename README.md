# GTK Transformer on FPGA with hls4ml

A hands-on tutorial that takes a trained Transformer for GigaTracker (GTK) track reconstruction and runs part of it on an Alveo U55C FPGA.

For each event the model reads the hits (x, y, GTK station, time). For every pair of hits *(i, j)* with *i ≤ j* it gives the probability that they are consecutive hits of the same particle. Two Transformer layers let each hit attend to all the others before the pairs are scored.

## The idea

[hls4ml](https://fastmachinelearning.org/hls4ml/) cannot convert the full model, because the encoder uses operations it doesn't support (for example `LayerNormalization` and attention). The pair classifier, a small MLP, converts fine. So the model is split:

| Part | Runs on |
|------|---------|
| Encoder (Transformer layers) | CPU, ONNX Runtime |
| Pair building (embeddings of hits *i* and *j* side by side) | FPGA, provided custom C++ kernel |
| Pair classifier (MLP, 64 → 2) | FPGA, converted with hls4ml |

## Layout

The notebooks come in two versions with the same names:

- `ex/` contains the exercises, with `# TODO` cells to complete.
- `sol/` contains the solutions.

| Notebook | Topic |
|----------|-------|
| `00_hls4ml_on_the_full_model` | Which layers hls4ml supports. Convert a small MLP, then try the full Transformer and see where it fails. |
| `01_onnx_graph_and_split` | Read the ONNX graph, find the cut points, run the model with ONNX Runtime, and split it into `encoder.onnx` and `pair_head.onnx`. Save the weights and value ranges. |
| `02_hls4ml_firmware` | Choose fixed-point precision, run C simulation, estimate resources and latency with Vitis HLS (the effect of the reuse factor), connect the custom kernel, and inspect the prebuilt bitstream. |
| `03_fpga_inference` | Load the bitstream on the U55C, run the hybrid model, check correctness against the C simulation, the float model and track reconstruction, and compare timing with the CPU. Speed it up with batching. |

Do them in order. Each one uses files produced by the one before.

## Requirements

- The project that provides `hi_common.py` and the `school/` folder. The notebooks look for them by walking up from the current directory.
- Conda environment `hybrid-tutorial`. Start Jupyter with `HI_CONDA_ENV=hybrid-tutorial source env.sh py`.
- Python packages: `hls4ml`, `onnx`, `onnxruntime`, `torch`, `numpy`, `matplotlib`.
- Exercises 0 and 1 need no Xilinx tools.
- Exercise 2 needs Vitis HLS for the synthesis steps. The full bitstream build takes about 2 hours, so a prebuilt `.xclbin` (49 MB) is downloaded from the course storage.
- Exercise 3 needs an Alveo U55C card and XRT.
