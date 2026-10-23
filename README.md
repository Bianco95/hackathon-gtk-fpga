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

## How the files fit together

The notebooks in `ex/` and `sol/` are the only entry points. Everything else in the top-level folder is support code and data that they load. The `ex/` and `sol/` notebooks share all of it: they differ only in the cells you have to complete.

Each notebook's setup cell walks up from the current directory until it finds a folder containing `utils.py` and `hi_common.py`. That folder becomes `ROOT_DIR` and is added to `sys.path`. This is why the notebooks must be opened from inside `ex/` or `sol/`, and why these files have to stay in the top-level folder.

### Support code

| File | Role | Used by |
|------|------|---------|
| `utils.py` | The helper library of the notebooks, loaded with `from utils import *`. It holds the course settings (`course_config`, `course_model_config`), all the plots, the bitstream download (`download_bitstream`, `PREBUILT_BITSTREAM`), the Vitis HLS report parser, the card check, the timing helpers, and the helpers for unpacking the kernel's candidates. | All four notebooks, directly. |
| `hi_common.py` | The shared project helpers: loading events (`load_events`), the checkpoint (`load_checkpoint`) and the PyTorch model (`build_models`), pair indexing, ONNX-to-MLP weight extraction (`mlp_layers_from_onnx`), and packing and unpacking of the kernel's candidates. | Notebooks 0, 1 and 2 import it as `hi_common`. Notebook 3 uses it only indirectly, through `utils.py`. |
| `gtk_tracking.py` | The GTK code copied from the tracking project: the event loader (`TransformerDataset`), the Transformer classes (`Encoder`, `EncoderONNX`), and the hit clustering and track reconstruction. | Never imported by a notebook. `hi_common.py` loads it for the events and the model, and `hybrid.py` calls its tracking functions. |
| `hybrid.py` | Only its tracking helpers are used here: `cluster_event` (cluster the hits as the model sees them) and `tracks_from_edges` (turn the predicted edges into tracks and compare them with the truth). | Not imported by a notebook. `utils.py` calls it in `load_test_events` (notebooks 1 and 3) and in `count_tracks` (notebook 3). |
| `pairhead_backends.py` | Python wrappers around the pair head. Only `PairEdgesCSim` is used: it loads the compiled C simulation of the custom kernel (`libpairedges_csim.so`) with `ctypes`. | Notebook 2, to check the kernel against the hls4ml model bit for bit. |

### Kernel, model and data

| Path | What it is | Used by |
|------|-----------|---------|
| `custom_kernel/pairedges_fact_kernel.cpp` | The Vitis HLS source of the `pairedges` kernel. It builds the pairs from the hit embeddings, runs the hls4ml pair classifier on them and returns only the candidate edges. | Notebook 2 copies it next to the generated hls4ml firmware, compiles its C simulation and synthesizes it. The prebuilt bitstream used in notebook 3 was built from it. |
| `model/full_current.onnx` | The trained full model, exported to ONNX. | Notebook 0 (try hls4ml on it), notebook 1 (inspect and split it), notebook 3 (CPU reference). |
| `model/tutorial_small_data.ckpt` | The PyTorch checkpoint of the same model. | Notebook 0, to get the trained weights and the PyTorch model. |
| `data/small_data/` | A small set of events with `features/`, `gt_tensors/` and `dataframe/` folders, each split into `validation` and `test`. | All notebooks, through `hi_common.load_events` or `utils.load_test_events`. Validation events calibrate the value ranges, and test events measure accuracy and tracks. |
| `work/` | Created by the notebooks, not part of the repository: `onnx/` (notebook 1), `data/` (notebook 1), `hls/` and `kernel/` (notebook 2), `fpga/` (the downloaded bitstream), `figures/`. | Each notebook reads what the previous ones wrote here. |

### What each notebook uses

| Notebook | Support code it uses | Reads | Writes |
|----------|---------------------|-------|--------|
| `00_hls4ml_on_the_full_model` | `utils.py`, `hi_common.py` (via it `gtk_tracking.py`) | `model/` (ONNX and checkpoint), `data/` | Nothing needed later. |
| `01_onnx_graph_and_split` | `utils.py`, `hi_common.py`, and `hybrid.py` and `gtk_tracking.py` through `load_test_events` | `model/full_current.onnx`, `data/` | `work/onnx/encoder.onnx` and `pair_head.onnx`, `work/data/` (classifier weights, and the pairs and logits of the validation and test events). |
| `02_hls4ml_firmware` | `utils.py`, `hi_common.py`, `pairhead_backends.py` (`PairEdgesCSim`), `custom_kernel/` | `work/data/`, `work/onnx/encoder.onnx`, `data/` | `work/hls/` (hls4ml projects), `work/kernel/` (firmware, kernel, C simulation library), `work/fpga/pairedges.xclbin` (the downloaded bitstream). |
| `03_fpga_inference` | `utils.py` directly (through it `hi_common.py`, `hybrid.py`, `gtk_tracking.py`), and `pynq` for the card | `model/full_current.onnx`, `work/onnx/encoder.onnx`, `work/fpga/pairedges.xclbin`, `data/` | `work/figures/`. |

Notebook 3 does not use `pairhead_backends.py` or the `HybridPipeline` of `hybrid.py`. It talks to the card directly with PYNQ, which is the point of the exercise.

### Parts that are not used

`pairhead_backends.py` and `hybrid.py` also contain code from the project they were taken from: `PairHeadCPU`, `PairHeadFPGA`, `PairHeadCSim`, `PairEdgesFPGA`, `HybridPipeline`, `OnCardPipeline` and `make_pair_head`. No notebook uses them. The FPGA classes need a host library (`fpga_host/build/libpairhead_xrt.so`) and a project configuration file that are not in this repository, so they cannot run here.

## Requirements

- The files in the top-level folder (`utils.py`, `hi_common.py`, `gtk_tracking.py`, `hybrid.py`, `pairhead_backends.py`) and the `custom_kernel/`, `model/` and `data/` folders, as described above. Keep the notebooks inside `ex/` or `sol/` so that they find them.
- Conda environment `hybrid-tutorial`. Start Jupyter with `HI_CONDA_ENV=hybrid-tutorial source env.sh py`.
- Python packages: `hls4ml`, `onnx`, `onnxruntime`, `torch`, `numpy`, `pandas`, `pyyaml`, `matplotlib`, and `pynq` for exercise 3.
- Exercises 0 and 1 need no Xilinx tools.
- Exercise 2 needs Vitis HLS for the synthesis steps. The full bitstream build takes about 2 hours, so a prebuilt `.xclbin` (49 MB) is downloaded from the course storage.
- Exercise 3 needs an Alveo U55C card and XRT.
