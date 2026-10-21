"""Useful methods for the school notebooks: the plots of the exercises and a few helpers.

The notebooks load them with `from utils import *` (only the names in __all__ are imported).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

__all__ = ["use_course_style", "print_dict", "plot_toolchain", "plot_layer_catalogue", "plot_operation_counts",
           "model_parts", "plot_where_unsupported", "load_hits_per_event", "plot_work_share", "plot_plan",
           "show_hls_layers", "encoder_multiply_adds", "classifier_multiply_adds",
           # exercise 1
           "shape_of", "softmax", "load_test_events", "plot_cuts", "plot_parameters_and_work",
           "plot_probability_matrix", "plot_split_check", "plot_value_ranges",
           # exercise 2
           "output_to_log_file", "U55C_RESOURCES", "read_csynth_report", "read_kernel_utilization", "fetch_prebuilt",
           "plot_bit_layout", "plot_precision_scan", "plot_fixed_point_error", "plot_synthesis_variants",
           "plot_data_per_event", "plot_kernel_pipeline", "plot_estimate_vs_routed",
           # exercise 3
           "check_card_is_free", "PATH_COLOR", "PATH_LABEL", "plot_agreement", "plot_latency", "plot_stages",
           "plot_card_call", "plot_batching", "plot_threads", "plot_summary"]

# ---------------------------------------------------------------------------------------------------------
# Colours and style
# ---------------------------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL_TILE = "#e9e8e2"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD = "#0ca30c"            # status colours: always shown with a ✓ / ✗ and a label
CRITICAL = "#d03b3b"
CPU_COLOR = CATEGORICAL[0]  # blue: runs on the CPU (same convention as the tutorial figures)
FPGA_COLOR = CATEGORICAL[1]  # orange: runs on the FPGA


def use_course_style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "figure.dpi": 110, "font.size": 10, "text.color": INK,
                         "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
                         "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
                         "axes.grid": False, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
                         "legend.frameon": False})


def tint(color, amount):
    # the colour mixed with the chart surface: amount 0 = surface, 1 = the colour itself
    color_rgb = np.array(to_rgb(color))
    surface_rgb = np.array(to_rgb(SURFACE))
    return tuple(surface_rgb + amount * (color_rgb - surface_rgb))


def _save(figure, save_to):
    if save_to is not None:
        figure.savefig(save_to, bbox_inches="tight")


def _box(axes, x, y, width, height, face, edge="none", line_width=0, rounding=0.06):
    axes.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                  facecolor=face, edgecolor=edge, linewidth=line_width))


# ---------------------------------------------------------------------------------------------------------
# 1 · The hls4ml tool chain
# ---------------------------------------------------------------------------------------------------------
def plot_toolchain(save_to=None):
    steps = [("Trained model", "PyTorch, ONNX\nor Keras"),
             ("hls4ml frontend", "reads the model,\nlayer by layer"),
             ("hls4ml backend", "writes C++ for each\nlayer, in fixed point"),
             ("Vitis HLS", "turns the C++ into\na hardware design"),
             ("Vivado / v++", "places and routes it:\nthe FPGA bitstream")]
    figure, axes = plt.subplots(figsize=(12, 2.6))
    box_width, gap = 2.0, 0.55
    for number, (title, detail) in enumerate(steps):
        x = number * (box_width + gap)
        highlighted = number == 1
        _box(axes, x, 0, box_width, 1.3, tint(FPGA_COLOR, 0.16) if highlighted else NEUTRAL_TILE,
             edge=FPGA_COLOR if highlighted else "none", line_width=2 if highlighted else 0, rounding=0.12)
        axes.text(x + 0.15, 1.05, f"{number + 1}", fontsize=9, color=INK_MUTED, va="center")
        axes.text(x + box_width / 2, 0.86, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        axes.text(x + box_width / 2, 0.38, detail, ha="center", va="center", fontsize=8.5, color=INK_SECONDARY)
        if number < len(steps) - 1:
            axes.add_patch(FancyArrowPatch((x + box_width + 0.06, 0.65), (x + box_width + gap - 0.06, 0.65),
                                           arrowstyle="-|>", mutation_scale=12, color=INK_MUTED, linewidth=1.2))
    axes.text(1 * (box_width + gap) + box_width / 2, -0.32, "this notebook: does step 2 work for our model?",
              ha="center", va="center", fontsize=9, color=INK_SECONDARY, style="italic")
    axes.set_xlim(-0.1, len(steps) * (box_width + gap) - gap + 0.1)
    axes.set_ylim(-0.55, 1.45)
    axes.axis("off")
    axes.set_title("From a trained network to FPGA firmware with hls4ml", pad=10)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


# ---------------------------------------------------------------------------------------------------------
# Exercise 1 · the catalogue of supported layers
# ---------------------------------------------------------------------------------------------------------
LAYER_CATEGORIES = [
    ("Dense / matrix", ["MatMul", "Linear", "einsum"]),
    ("Convolution & pooling", ["Conv", "Conv1d", "Conv2d", "AveragePool", "MaxPool", "GlobalAveragePool",
                               "GlobalMaxPool", "AvgPool1d", "AvgPool2d", "MaxPool1d", "MaxPool2d"]),
    ("Activation", ["Relu", "ReLU", "Elu", "ELU", "LeakyRelu", "LeakyReLU", "PRelu", "PReLU", "Selu", "Sigmoid",
                    "Softmax", "Softplus", "Softsign", "Tanh", "ThresholdedRelu", "Threshold"]),
    ("Normalisation", ["BatchNormalization", "BatchNorm1d", "BatchNorm2d", "Batch_norm", "LayerNorm",
                       "LayerNormalization"]),
    ("Arithmetic & merge", ["Add", "Sub", "Mul", "Div", "Max", "Min", "Sum", "Average", "Concat", "add", "sub",
                            "subtract", "mul", "multiply", "fmax", "fmin", "maximum", "minimum", "cat", "concat",
                            "concatenate"]),
    ("Shape & resize", ["Reshape", "Flatten", "Transpose", "Resize", "View", "squeeze", "unsqueeze", "Upsample",
                        "UpsamplingBilinear2d", "UpsamplingNearest2d", "ConstantPad1d", "ConstantPad2d", "Constant"]),
    ("Recurrent", ["GRU", "LSTM", "RNN"]),
    ("Quantisation", ["Quant", "IntQuant", "BipolarQuant"]),
]


def _category_of(layer_name):
    for number, (category, members) in enumerate(LAYER_CATEGORIES):
        if layer_name in members:
            return number
    return None                     # a layer added in a newer hls4ml: shown in an "Other" column


def plot_layer_catalogue(onnx_layers, pytorch_layers, save_to=None):
    if onnx_layers is None or pytorch_layers is None:
        print("onnx_layers or pytorch_layers is still None: complete the task that defines them first.")
        return
    panels = [("ONNX frontend", sorted(onnx_layers)), ("PyTorch frontend", sorted(pytorch_layers, key=str.lower))]
    categories = [category for category, _ in LAYER_CATEGORIES]
    colors = list(CATEGORICAL)
    columns_per_panel = []
    for _, layers in panels:
        columns = [[] for _ in categories]
        others = []
        for layer in layers:
            number = _category_of(layer)
            if number is None:
                others.append(layer)
            else:
                columns[number].append(layer)
        columns_per_panel.append(columns + ([others] if others else []))
    if any(len(columns) > len(categories) for columns in columns_per_panel):
        categories = categories + ["Other"]
        colors = colors + [INK_MUTED]
        columns_per_panel = [columns + [[]] * (len(categories) - len(columns)) for columns in columns_per_panel]

    rows_per_panel = [max(1, max(len(column) for column in columns)) for columns in columns_per_panel]
    figure, axes_list = plt.subplots(2, 1, figsize=(1.62 * len(categories), 0.3 * (sum(rows_per_panel) + 7)),
                                     gridspec_kw={"height_ratios": [rows + 2.2 for rows in rows_per_panel]})
    for axes, (panel_title, layers), columns, n_rows in zip(axes_list, panels, columns_per_panel, rows_per_panel):
        for column_number, (category, color, members) in enumerate(zip(categories, colors, columns)):
            x = column_number
            axes.text(x + 0.05, 0.55, category, fontsize=8.5, fontweight="bold", color=INK_SECONDARY, va="center")
            axes.plot([x + 0.05, x + 0.95], [0.85, 0.85], color=color, linewidth=2.5, solid_capstyle="round")
            if not members:
                axes.text(x + 0.12, 1.5, "none", fontsize=8, color=INK_MUTED, style="italic", va="center")
            for row, layer in enumerate(members):
                y = 1.1 + row
                _box(axes, x + 0.05, y + 0.06, 0.9, 0.8, tint(color, 0.18))
                axes.text(x + 0.12, y + 0.46, layer, fontsize=8, color=INK, va="center")
        axes.set_xlim(0, len(categories))
        axes.set_ylim(n_rows + 1.3, -0.4)
        axes.axis("off")
        axes.set_title(f"{panel_title} · {len(layers)} layers", fontsize=11, pad=4)
    figure.suptitle("What hls4ml can read", x=0.01, ha="left", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


# ---------------------------------------------------------------------------------------------------------
# Exercise 3 · the operations of the GTK Transformer
# ---------------------------------------------------------------------------------------------------------
SKIPPED_BY_HLS4ML_ONNX = ["Dropout", "Identity"]       # accepted and removed by the ONNX converter


def plot_operation_counts(operation_counts, onnx_layers, save_to=None):
    if operation_counts is None or onnx_layers is None:
        print("onnx_layers or operation_counts is still None: complete the tasks that define them first.")
        return
    supported = set(onnx_layers) | set(SKIPPED_BY_HLS4ML_ONNX)
    ordered = sorted(operation_counts.items(), key=lambda item: (-item[1], item[0]))
    names = [name for name, _ in ordered]
    counts = [count for _, count in ordered]
    is_supported = [name in supported for name in names]

    figure, axes = plt.subplots(figsize=(8.5, 0.32 * len(names) + 1.4))
    positions = np.arange(len(names))
    axes.barh(positions, counts, height=0.62, color=[GOOD if ok else CRITICAL for ok in is_supported])
    for position, count, ok in zip(positions, counts, is_supported):
        label = f"{count}" if ok else f"{count}   ✗ not supported"
        axes.text(count + 0.4, position, label, va="center", fontsize=8.5, color=INK_SECONDARY)
    axes.set_yticks(positions)
    axes.set_yticklabels(names)
    axes.invert_yaxis()
    axes.set_xlabel("nodes in the graph")
    axes.xaxis.grid(True)
    axes.set_axisbelow(True)
    axes.set_xlim(0, max(counts) * 1.45)
    axes.tick_params(axis="y", length=0)
    n_bad = sum(count for count, ok in zip(counts, is_supported) if not ok)
    axes.set_title(f"The GTK Transformer: {sum(counts)} nodes, {len(names)} operation types, "
                   f"{n_bad} nodes hls4ml cannot read")
    axes.legend(handles=[Patch(color=GOOD, label="✓ supported by hls4ml's ONNX frontend"),
                         Patch(color=CRITICAL, label="✗ not supported")], loc="lower right", fontsize=8.5)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def model_parts(model):
    """{node name: "encoder" | "pair building" | "pair classifier"}, found from the graph:
    the encoder is everything the last LayerNormalization needs (the hit embeddings), the pair building is what
    the Concat needs on top of that, the pair classifier is the rest."""
    producer_of = {}
    for node in model.graph.node:
        for output in node.output:
            producer_of[output] = node

    def nodes_needed_for(tensor_name):
        needed = set()
        to_visit = [tensor_name]
        while to_visit:
            node = producer_of.get(to_visit.pop())
            if node is not None and node.name not in needed:
                needed.add(node.name)
                to_visit.extend(node.input)
        return needed

    last_layer_norm = [node for node in model.graph.node if node.op_type == "LayerNormalization"][-1]
    concat_node = [node for node in model.graph.node if node.op_type == "Concat"][0]
    encoder = nodes_needed_for(last_layer_norm.output[0])
    pair_building = nodes_needed_for(concat_node.output[0]) - encoder
    parts = {}
    for node in model.graph.node:
        if node.name in encoder:
            parts[node.name] = "encoder"
        elif node.name in pair_building:
            parts[node.name] = "pair building"
        else:
            parts[node.name] = "pair classifier"
    return parts


PART_DESCRIPTIONS = {"encoder": "Encoder · embedding + 2 Transformer layers",
                     "pair building": "Pair building · picks the embeddings of hits i and j",
                     "pair classifier": "Pair classifier · an MLP on each pair, then the softmax"}


def plot_where_unsupported(model, onnx_layers, save_to=None):
    if onnx_layers is None:
        print("onnx_layers is still None: complete the task that defines it first.")
        return
    supported = set(onnx_layers) | set(SKIPPED_BY_HLS4ML_ONNX)
    parts = model_parts(model)
    per_row = 8
    rows = []                                    # ("header", text) or ("tiles", [op types])
    for part in ("encoder", "pair building", "pair classifier"):
        operations = [node.op_type for node in model.graph.node if parts[node.name] == part]
        n_bad = sum(operation not in supported for operation in operations)
        status = f"{n_bad} not supported" if n_bad else "all supported ✓"
        rows.append(("header", f"{PART_DESCRIPTIONS[part]}   —   {len(operations)} nodes, {status}"))
        for start in range(0, len(operations), per_row):
            rows.append(("tiles", operations[start:start + per_row]))

    figure, axes = plt.subplots(figsize=(12, 0.34 * len(rows) + 1.0))
    for row_number, (kind, content) in enumerate(rows):
        y = row_number
        if kind == "header":
            axes.text(0.02, y + 0.62, content, fontsize=9.5, fontweight="bold", color=INK_SECONDARY, va="center")
            continue
        for column, operation in enumerate(content):
            ok = operation in supported
            _box(axes, column + 0.03, y + 0.08, 0.94, 0.8, NEUTRAL_TILE if ok else CRITICAL, rounding=0.1)
            label = operation if ok else f"✗ {operation}"
            axes.text(column + 0.5, y + 0.49, label, fontsize=7.2 if len(label) > 14 else 8, ha="center",
                      va="center", color=INK if ok else "white", fontweight="normal" if ok else "bold")
    axes.set_xlim(0, per_row)
    axes.set_ylim(len(rows) + 0.1, -0.1)
    axes.axis("off")
    axes.set_title("Where hls4ml gets stuck: every node of the graph, in order, by part of the model", pad=8)
    axes.legend(handles=[Patch(color=NEUTRAL_TILE, label="✓ supported"), Patch(color=CRITICAL, label="✗ not supported")],
                loc="upper right", bbox_to_anchor=(1.0, 1.06), ncol=2, fontsize=8.5)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


# ---------------------------------------------------------------------------------------------------------
# After Exercise 6 · where the work is, and the plan
# ---------------------------------------------------------------------------------------------------------
D_MODEL, FEED_FORWARD, N_LAYERS = 32, 64, 2       # the GTK Transformer's sizes


def encoder_multiply_adds(n_hits):
    # multiply-adds of the encoder for one event of n_hits hits (same formula as tutorial notebook 1)
    qkv_projections = 3 * n_hits * D_MODEL * D_MODEL
    attention = 2 * n_hits * n_hits * D_MODEL
    output_projection = n_hits * D_MODEL * D_MODEL
    feed_forward = 2 * n_hits * D_MODEL * FEED_FORWARD
    embedding = n_hits * 4 * D_MODEL
    return embedding + N_LAYERS * (qkv_projections + attention + output_projection + feed_forward)


def classifier_multiply_adds(n_hits):
    n_pairs = n_hits * (n_hits + 1) // 2
    return n_pairs * (64 * 64 + 64 * 2)


def load_test_events(config, model_config):
    """The test events after the repo's clustering (as the model sees them): [{"name", "features" [N, 4]}]."""
    import hi_common
    from hybrid import cluster_event
    tracking = hi_common.import_metrics(config)
    events = []
    for event in hi_common.load_events(config, model_config, config["data"]["eval_split"]):
        _, clustered_hits, _ = cluster_event(tracking, event["df"], event["features"])
        if len(clustered_hits) > 0:
            events.append({"name": event["name"], "features": clustered_hits})
    return events


def load_hits_per_event(config, model_config):
    """Number of hits of each test event, after the repo's clustering."""
    return np.array([len(event["features"]) for event in load_test_events(config, model_config)])


def plot_work_share(hits_per_event, save_to=None):
    n_grid = np.arange(2, hits_per_event.max() + 1)
    median = int(np.median(hits_per_event))
    share = classifier_multiply_adds(median) / (classifier_multiply_adds(median) + encoder_multiply_adds(median))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    encoder_work = encoder_multiply_adds(n_grid) / 1e6
    classifier_work = classifier_multiply_adds(n_grid) / 1e6
    top.plot(n_grid, classifier_work, color=FPGA_COLOR, linewidth=2)
    top.plot(n_grid, encoder_work, color=CPU_COLOR, linewidth=2)
    top.text(n_grid[-1] + 1.5, classifier_work[-1], "pair classifier\n(hls4ml can convert it)", va="center",
             fontsize=8.5, color=INK_SECONDARY)
    top.text(n_grid[-1] + 1.5, encoder_work[-1], "encoder", va="center", fontsize=8.5, color=INK_SECONDARY)
    top.axvline(median, color=INK_MUTED, linestyle=":", linewidth=1.2)
    top.annotate(f"median event: {median} hits, {median * (median + 1) // 2:,} pairs\n"
                 f"→ {share:.0%} of the multiply-adds are in the pair classifier",
                 xy=(median, classifier_multiply_adds(median) / 1e6), xytext=(median + 12, classifier_work[-1] * 0.62),
                 fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    top.plot([median], [classifier_multiply_adds(median) / 1e6], "o", color=FPGA_COLOR, markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=1.5)
    top.set_ylabel("million multiply-adds per event")
    top.yaxis.grid(True)
    top.set_axisbelow(True)
    top.set_title("Where the work is: the pair classifier runs once per pair, K = N(N+1)/2 times per event")

    bottom.hist(hits_per_event, bins=np.arange(0, hits_per_event.max() + 6, 5), color="#a9a79f",
                edgecolor=SURFACE, linewidth=1.5)
    bottom.axvline(median, color=INK_MUTED, linestyle=":", linewidth=1.2)
    bottom.set_xlabel("hits per event, N")
    bottom.set_ylabel("test events")
    bottom.yaxis.grid(True)
    bottom.set_axisbelow(True)
    bottom.set_xlim(0, n_grid[-1] + 24)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()
    return share


def plot_plan(save_to=None):
    parts = [("Encoder", "embedding +\n2 Transformer layers", False, CPU_COLOR, "CPU · ONNX Runtime",
              "you split it off the model (exercise 1)"),
             ("Pair building", "picks the embeddings\nof hits i and j", False, FPGA_COLOR, "FPGA · custom C++ kernel",
              "provided: already written for you"),
             ("Pair classifier", "Linear → ReLU → Linear", True, FPGA_COLOR, "FPGA · hls4ml",
              "your focus: the hls4ml conversion (exercise 2)")]
    figure, axes = plt.subplots(figsize=(11, 3.3))
    box_width, gap = 3.0, 0.7
    for number, (title, detail, convertible, color, destination, who) in enumerate(parts):
        x = number * (box_width + gap)
        _box(axes, x, 0.55, box_width, 1.35, tint(color, 0.15), edge=color, line_width=1.8, rounding=0.12)
        axes.text(x + box_width / 2, 1.58, title, ha="center", va="center", fontsize=11, fontweight="bold")
        axes.text(x + box_width / 2, 1.08, detail, ha="center", va="center", fontsize=8.8, color=INK_SECONDARY)
        icon, icon_color = ("✓", GOOD) if convertible else ("✗", CRITICAL)
        axes.text(x + 0.18, 1.72, icon, fontsize=13, color=icon_color, fontweight="bold", va="center")
        axes.text(x + box_width / 2, 0.68, "hls4ml can convert it" if convertible else "hls4ml cannot convert it",
                  ha="center", va="center", fontsize=8, color=INK_SECONDARY, style="italic")
        axes.text(x + box_width / 2, 0.22, f"→ {destination}", ha="center", va="center", fontsize=9.5,
                  fontweight="bold", color=INK)
        axes.text(x + box_width / 2, -0.12, who, ha="center", va="center", fontsize=8.5, color=INK_SECONDARY,
                  style="italic")
        if number < len(parts) - 1:
            axes.add_patch(FancyArrowPatch((x + box_width + 0.08, 1.22), (x + box_width + gap - 0.08, 1.22),
                                           arrowstyle="-|>", mutation_scale=14, color=INK_MUTED, linewidth=1.3))
    axes.set_xlim(-0.1, len(parts) * (box_width + gap) - gap + 0.1)
    axes.set_ylim(-0.32, 2.1)
    axes.axis("off")
    axes.set_title("The plan: split the model, and give each part to the tool that can run it", pad=6)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


# ---------------------------------------------------------------------------------------------------------
# Small helpers used by the notebook
# ---------------------------------------------------------------------------------------------------------
def show_hls_layers(hls_model):
    if hls_model is None:
        print("The hls4ml model is still None: complete the conversion first.")
        return
    print(f"{'layer':8s} {'type':11s} fixed-point types")
    for layer in hls_model.get_layers():
        types = ", ".join(f"{kind.replace('_t', '')}: {variable.precision}" for kind, variable in layer.types.items())
        print(f"{layer.name:8s} {layer.class_name:11s} {types}")


def print_dict(dictionary, indent=0):
    # print a (nested) configuration dictionary, one key per line
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")



# ---------------------------------------------------------------------------------------------------------
# Exercise 1 · the ONNX graph and the split
# ---------------------------------------------------------------------------------------------------------
def shape_of(value_info):
    # the shape of an ONNX tensor: a number for a fixed size, a name for a dynamic one
    return [axis.dim_param or axis.dim_value for axis in value_info.type.tensor_type.shape.dim]


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def plot_cuts(save_to=None):
    # the model as a chain of parts, with the two cuts and where each part runs in the hybrid
    blocks = [("Encoder", "embedding +\n2 Transformer layers", CPU_COLOR, "CPU · encoder.onnx"),
              ("Pair building", "e[i] and e[j]\nside by side", FPGA_COLOR, "FPGA · custom kernel (provided)"),
              ("Pair classifier", "Linear → ReLU →\nLinear", FPGA_COLOR, "FPGA · hls4ml · pair_head.onnx"),
              ("Softmax", "probabilities\n[1, K, 2]", CPU_COLOR, "CPU")]
    tensors = ["data\n[N, 4]", "embeddings\n[1, N, 32]", "pair features\n[1, K, 64]", "logits\n[1, K, 2]"]
    cuts = {1: "cut 1", 2: "cut 2"}                 # cut before block 1 and before block 2
    figure, axes = plt.subplots(figsize=(12.5, 3.6))
    box_width, gap = 2.2, 1.15
    x0 = 1.05
    axes.text(0.45, 1.25, tensors[0], ha="center", va="center", fontsize=8.5, family="monospace", color=INK_SECONDARY)
    for number, (title, detail, color, destination) in enumerate(blocks):
        x = x0 + number * (box_width + gap)
        _box(axes, x, 0.7, box_width, 1.1, tint(color, 0.15), edge=color, line_width=1.6, rounding=0.1)
        axes.text(x + box_width / 2, 1.5, title, ha="center", va="center", fontsize=10.5, fontweight="bold")
        axes.text(x + box_width / 2, 1.0, detail, ha="center", va="center", fontsize=8.3, color=INK_SECONDARY)
        axes.text(x + box_width / 2, 0.38, destination, ha="center", va="center", fontsize=8.3, color=INK_SECONDARY,
                  style="italic")
        start = (x0 - 0.15, 1.25) if number == 0 else (x - gap + 0.06, 1.25)
        axes.add_patch(FancyArrowPatch(start, (x - 0.04, 1.25), arrowstyle="-|>", mutation_scale=12,
                                       color=INK_MUTED, linewidth=1.2))
        if number > 0:
            middle = x - gap / 2
            axes.text(middle, 1.62, tensors[number], ha="center", va="center", fontsize=8, family="monospace",
                      color=INK_SECONDARY, zorder=3,
                      bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
            if number in cuts:
                axes.plot([middle, middle], [0.15, 2.05], color=CRITICAL, linestyle="--", linewidth=1.4)
                axes.text(middle, 2.18, f"✂ {cuts[number]}", ha="center", va="center", fontsize=9,
                          fontweight="bold", color=INK)
    pair_x = x0 + 1 * (box_width + gap) + box_width / 2
    axes.text(pair_x, 2.55, "pair_indices [K, 2]", ha="center", va="center", fontsize=8, family="monospace",
              color=INK_SECONDARY)
    axes.add_patch(FancyArrowPatch((pair_x, 2.42), (pair_x, 1.84), arrowstyle="-|>", mutation_scale=12,
                                   color=INK_MUTED, linewidth=1.2))
    axes.set_xlim(0, x0 + len(blocks) * (box_width + gap) - gap + 0.1)
    axes.set_ylim(0.1, 2.75)
    axes.axis("off")
    axes.set_title("The trained model, the two cuts, and where each part runs in the hybrid", pad=4)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_parameters_and_work(encoder_parameters, classifier_parameters, n_hits, save_to=None):
    encoder_work = encoder_multiply_adds(n_hits)
    classifier_work = classifier_multiply_adds(n_hits)
    rows = [("parameters\n(numbers learned in training)", encoder_parameters, classifier_parameters),
            (f"multiply-adds for an event\nof {n_hits} hits", encoder_work, classifier_work)]
    figure, axes = plt.subplots(figsize=(10, 2.8))
    for row, (label, encoder_value, classifier_value) in enumerate(rows):
        total = encoder_value + classifier_value
        encoder_share = encoder_value / total
        axes.barh(row, encoder_share, color=CPU_COLOR, height=0.55, edgecolor=SURFACE, linewidth=2)
        axes.barh(row, 1 - encoder_share, left=encoder_share, color=FPGA_COLOR, height=0.55, edgecolor=SURFACE,
                  linewidth=2)
        axes.text(encoder_share / 2, row, f"encoder\n{encoder_share:.0%} · {encoder_value:,.0f}", ha="center",
                  va="center", fontsize=8.5, color="white", fontweight="bold")
        axes.text(encoder_share + (1 - encoder_share) / 2, row,
                  f"pair classifier\n{1 - encoder_share:.0%} · {classifier_value:,.0f}", ha="center", va="center",
                  fontsize=8.5, color="white", fontweight="bold")
    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row[0] for row in rows])
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xticks([])
    axes.tick_params(axis="y", length=0)
    for side in ("bottom", "left"):
        axes.spines[side].set_visible(False)
    axes.set_title("The pair classifier: a small part of the parameters, most of the work")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


PROBABILITY_COLORMAP = LinearSegmentedColormap.from_list(
    "probability", ["#f3f7fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def plot_probability_matrix(probabilities, pairs, threshold, title="", save_to=None):
    """probabilities [K, 2] for the pairs [K, 2] of one event: p1 drawn at (i, j) of an N x N matrix."""
    n_hits = int(pairs.max()) + 1
    matrix = np.full((n_hits, n_hits), np.nan)
    matrix[pairs[:, 0], pairs[:, 1]] = probabilities[:, 1]
    top = np.ceil(probabilities[:, 1].max() * 10) / 10          # colour scale up to the largest p1, rounded up
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
    image = left.imshow(np.ma.masked_invalid(matrix), cmap=PROBABILITY_COLORMAP, vmin=0, vmax=top,
                        interpolation="nearest")
    left.set_xlabel("hit j")
    left.set_ylabel("hit i")
    left.set_title(f"p₁ of every pair (i, j), i ≤ j{title}")
    for side in ("top", "right", "bottom", "left"):
        left.spines[side].set_visible(False)
    colorbar = figure.colorbar(image, ax=left, fraction=0.046, pad=0.03)
    colorbar.set_label("p₁: probability that (i, j) is an edge", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.axhline(threshold, color=CRITICAL, linewidth=1.5)

    p1 = probabilities[:, 1]
    right.hist(p1, bins=np.linspace(0, top, 31), color=CPU_COLOR, edgecolor=SURFACE, linewidth=1)
    right.axvline(threshold, color=CRITICAL, linestyle="--", linewidth=1.4)
    n_edges = int((p1 > threshold).sum())
    right.set_yscale("log")
    right.text(threshold + 0.03, 0.9, f"threshold {threshold}\n{n_edges} of {len(p1):,} pairs above it",
               transform=right.get_xaxis_transform(), fontsize=8.5, color=INK, va="top")
    right.set_xlabel("p₁")
    right.set_ylabel("pairs")
    right.yaxis.grid(True)
    right.set_axisbelow(True)
    right.set_title("Distribution of p₁")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_split_check(full_p1, split_p1, save_to=None, max_points=20000):
    differences = np.abs(split_p1 - full_p1)
    rng = np.random.default_rng(0)
    shown = rng.choice(len(full_p1), size=min(max_points, len(full_p1)), replace=False)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    top = np.ceil(max(full_p1.max(), split_p1.max()) * 10) / 10
    left.plot([0, top], [0, top], color=BASELINE, linewidth=1)
    left.scatter(full_p1[shown], split_p1[shown], s=8, color=CPU_COLOR, alpha=0.35, edgecolors="none")
    left.set_xlabel("p₁ of the full model")
    left.set_ylabel("p₁ of the split model")
    left.set_title(f"Split vs full model ({len(shown):,} random pairs)")
    left.set_xlim(-0.02, top)
    left.set_ylim(-0.02, top)
    left.set_aspect("equal")
    left.grid(True)
    left.set_axisbelow(True)

    exact = int((differences == 0).sum())
    nonzero = differences[differences > 0]
    if len(nonzero):
        bins = np.logspace(np.floor(np.log10(nonzero.min())), np.ceil(np.log10(nonzero.max())), 30)
        right.hist(nonzero, bins=bins, color=CPU_COLOR, edgecolor=SURFACE, linewidth=1)
        right.set_xscale("log")
    right.set_xlabel("|p₁ split − p₁ full|  (pairs with a difference)")
    right.set_ylabel("pairs")
    right.yaxis.grid(True)
    right.set_axisbelow(True)
    right.set_title("How different?")
    right.text(0.02, 0.95, f"{exact / len(differences):.0%} of the {len(differences):,} pairs: exactly equal\n"
                           f"largest difference: {differences.max():.1e}",
               transform=right.transAxes, ha="left", va="top", fontsize=9, color=INK)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_value_ranges(pair_features, layers, save_to=None):
    """pair_features [M, 64] and layers {"W1", "b1", "W2", "b2"}: the numbers the FPGA has to represent."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    values = pair_features.ravel()
    largest = np.abs(values).max()
    left.hist(values, bins=120, color=CPU_COLOR, edgecolor="none")
    left.set_yscale("log")
    for side in (-largest, largest):
        left.axvline(side, color=INK_MUTED, linestyle=":", linewidth=1.2)
    left.text(0.98, 0.95, f"largest |value| {largest:.2f}", transform=left.transAxes, ha="right", va="top",
              fontsize=9)
    left.set_xlabel("value")
    left.set_ylabel("count")
    left.set_title(f"Pair features: the classifier's inputs ({pair_features.shape[0]:,} pairs × 64)")
    left.yaxis.grid(True)
    left.set_axisbelow(True)

    parameters = np.concatenate([layers[name].ravel() for name in ("W1", "b1", "W2", "b2")])
    right.hist(parameters, bins=80, color=FPGA_COLOR, edgecolor="none")
    right.text(0.98, 0.95, f"largest |value| {np.abs(parameters).max():.3f}", transform=right.transAxes,
               ha="right", va="top", fontsize=9)
    right.set_xlabel("value")
    right.set_ylabel("count")
    right.set_title(f"Weights and biases of the pair classifier ({len(parameters):,})")
    right.yaxis.grid(True)
    right.set_axisbelow(True)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()



# ---------------------------------------------------------------------------------------------------------
# Exercise 2 · hls4ml and the firmware
# ---------------------------------------------------------------------------------------------------------
import contextlib
import json
import os
import re
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path

U55C_RESOURCES = {"DSP": 9024, "LUT": 1303680, "FF": 2607360, "BRAM_18K": 4032, "URAM": 960}   # the whole FPGA


@contextlib.contextmanager
def output_to_log_file(log_path):
    """Send the output of a block to a log file. The Xilinx tools that hls4ml starts write to the process's
    file descriptors 1 and 2 directly, so those are redirected as well as Python's print."""
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_path, "w") as log_file:
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        try:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                yield
        finally:
            log_file.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def read_csynth_report(xml_path):
    """The numbers we need from a Vitis HLS synthesis report (<top>_csynth.xml)."""
    report = ElementTree.parse(xml_path).getroot()
    performance = report.find("PerformanceEstimates")
    def number(path):
        text = performance.find(path).text
        return int(text) if text.isdigit() else text        # "undef" when Vitis HLS cannot bound a loop

    summary = {"clock_ns": float(performance.find("SummaryOfTimingAnalysis/EstimatedClockPeriod").text),
               "latency": number("SummaryOfOverallLatency/Worst-caseLatency"),
               "interval": number("SummaryOfOverallLatency/Interval-max")}
    for resource in U55C_RESOURCES:
        element = report.find(f"AreaEstimates/Resources/{resource}")
        summary[resource] = int(element.text) if element is not None else 0
    return summary


def read_kernel_utilization(report_path):
    """The resources of the kernel after place and route, from Vivado's kernel utilization report."""
    for line in Path(report_path).read_text().splitlines():
        cells = line.split("|")
        if len(cells) > 2 and cells[1].strip() == "pairedges_1":
            numbers = [int(cell.split("[")[0]) for cell in cells[2:-1]]
            lut, _, registers, bram36, uram, dsp = numbers
            return {"DSP": dsp, "LUT": lut, "FF": registers, "BRAM_18K": 2 * bram36, "URAM": uram}
    raise ValueError(f"no pairedges_1 line in {report_path}")


PREBUILT_FILES = ["pairedges.json", "kernel_util_routed.rpt", "build_summary.txt"]   # the bitstream's description


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_prebuilt(bitstream_url, description_dir, destination):
    """Copy the description and the reports of the prebuilt bitstream from description_dir (school/prebuilt/),
    download the bitstream from bitstream_url as pairedges.xclbin (only if it is not already in destination),
    and check its SHA-256 against the one in the description."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in PREBUILT_FILES:
        shutil.copy(Path(description_dir) / name, destination / name)
        print(f"{name:25s} copied from {Path(description_dir).name}/")
    expected_sha256 = json.loads((destination / "pairedges.json").read_text())["xclbin_sha256"]

    bitstream = destination / "pairedges.xclbin"
    if bitstream.exists() and _sha256(bitstream) == expected_sha256:
        print(f"{'pairedges.xclbin':25s} already downloaded")
    else:
        print(f"{'pairedges.xclbin':25s} downloading from {bitstream_url} ...", flush=True)
        urllib.request.urlretrieve(bitstream_url, bitstream)
        if _sha256(bitstream) != expected_sha256:
            raise RuntimeError(f"the file downloaded from {bitstream_url} is not the expected bitstream (SHA-256)")
        print(f"{'pairedges.xclbin':25s} downloaded, {bitstream.stat().st_size / 1e6:.1f} MB, SHA-256 checked")
    files = {name: destination / name for name in PREBUILT_FILES}
    files["pairedges.xclbin"] = bitstream
    return files


def _fixed_bits(type_text):
    # "ap_fixed<14,4,AP_RND_CONV,AP_SAT>" -> (14, 4)
    total, integer = re.findall(r"\d+", type_text)[:2]
    return int(total), int(integer)


def plot_bit_layout(types, save_to=None):
    """One row per tensor: the bits of its ap_fixed type, sign / integer / fraction."""
    labels = {"input": "input (pair features)", "hidden": "hidden (fc1, ReLU)", "logits": "logits (fc2)",
              "w1": "fc1 weights", "b1": "fc1 biases", "w2": "fc2 weights", "b2": "fc2 biases"}
    colors = {"sign": CATEGORICAL[0], "integer": CATEGORICAL[2], "fraction": CATEGORICAL[1]}
    names = list(types)
    widest = max(_fixed_bits(types[name])[0] for name in names)
    figure, axes = plt.subplots(figsize=(12, 0.45 * len(names) + 1.3))
    for row, name in enumerate(names):
        total, integer = _fixed_bits(types[name])
        kinds = ["sign"] + ["integer"] * (integer - 1) + ["fraction"] * (total - integer)
        for bit, kind in enumerate(kinds):
            _box(axes, bit + 0.06, row + 0.12, 0.88, 0.76, tint(colors[kind], 0.75 if kind != "sign" else 1.0),
                 rounding=0.08)
        largest = 2.0 ** (integer - 1)
        axes.text(-0.4, row + 0.5, labels.get(name, name), ha="right", va="center", fontsize=9, color=INK)
        axes.text(widest + 0.6, row + 0.5, f"ap_fixed<{total},{integer}>   range ±{largest:g}, step 2⁻{total - integer}",
                  ha="left", va="center", fontsize=8.8, color=INK_SECONDARY, family="monospace")
    axes.set_xlim(-6.5, widest + 14)
    axes.set_ylim(len(names) + 0.1, -0.9)
    axes.axis("off")
    axes.legend(handles=[Patch(color=colors["sign"], label="sign bit"),
                         Patch(color=tint(colors["integer"], 0.75), label="integer bits"),
                         Patch(color=tint(colors["fraction"], 0.75), label="fraction bits")],
                loc="upper left", bbox_to_anchor=(0.18, 1.02), ncol=3, fontsize=8.5)
    axes.set_title("The bits of every number in the firmware: one square per bit", pad=14)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_precision_scan(scan_results, selected, max_dp_limit, save_to=None):
    """scan_results {(weight bits, data bits): {"max_abs_dp", "accepted", ...}}; selected = (weight, data) bits."""
    names = [f"w{weight}_d{data}" for weight, data in scan_results]
    values = [result["max_abs_dp"] for result in scan_results.values()]
    accepted = [result["accepted"] for result in scan_results.values()]
    figure, axes = plt.subplots(figsize=(9, 3.6))
    positions = np.arange(len(names))
    axes.bar(positions, values, width=0.6, color=[GOOD if ok else "#b5b3ab" for ok in accepted])
    for position, value, ok, bits in zip(positions, values, accepted, scan_results):
        is_selected = tuple(bits) == tuple(selected)
        label = "✓ selected" if is_selected else ("✓" if ok else "✗")
        axes.text(position, value * 0.55, label, ha="center", va="top", fontsize=9, color="white" if ok else INK,
                  fontweight="bold")
    axes.axhline(max_dp_limit, color=CRITICAL, linestyle="--", linewidth=1.2)
    axes.text(len(names) - 0.55, max_dp_limit * 1.12, "acceptance limit on max |Δp|", ha="right", va="bottom",
              fontsize=8.5, color=INK_SECONDARY)
    axes.set_yscale("log")
    axes.set_ylim(min(values) / 3, max(values) * 6)
    axes.set_xticks(positions)
    axes.set_xticklabels(names)
    axes.set_xlabel("configuration: weight bits (w) and data bits (d)")
    axes.set_ylabel("max |p₁ fixed − p₁ float|")
    axes.yaxis.grid(True)
    axes.set_axisbelow(True)
    axes.legend(handles=[Patch(color=GOOD, label="✓ accepted"), Patch(color="#b5b3ab", label="✗ rejected")],
                loc="upper right", fontsize=8.5)
    axes.set_title("Precision scan: how far the fixed-point probabilities move from the float ones")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_fixed_point_error(p_fixed, p_float, name, save_to=None):
    differences = p_fixed - p_float
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    left.hist(differences, bins=120, color=FPGA_COLOR, edgecolor="none")
    left.set_yscale("log")
    left.set_xlabel("p₁ (fixed point) − p₁ (float)")
    left.set_ylabel("pairs")
    left.yaxis.grid(True)
    left.set_axisbelow(True)
    left.set_title(f"{name}: error on {len(differences):,} test pairs")
    left.text(0.98, 0.95, f"mean |Δp| {np.abs(differences).mean():.1e}\nmax |Δp| {np.abs(differences).max():.1e}",
              transform=left.transAxes, ha="right", va="top", fontsize=9)

    rng = np.random.default_rng(0)
    shown = rng.choice(len(p_float), size=min(20000, len(p_float)), replace=False)
    top = np.ceil(max(p_float.max(), p_fixed.max()) * 10) / 10
    right.plot([0, top], [0, top], color=BASELINE, linewidth=1)
    right.scatter(p_float[shown], p_fixed[shown], s=6, color=FPGA_COLOR, alpha=0.35, edgecolors="none")
    right.set_xlim(-0.02, top)
    right.set_ylim(-0.02, top)
    right.set_aspect("equal")
    right.grid(True)
    right.set_axisbelow(True)
    right.set_xlabel("p₁ float (PyTorch)")
    right.set_ylabel("p₁ fixed point (hls4ml)")
    right.set_title(f"Fixed point vs float ({len(shown):,} random pairs)")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_synthesis_variants(results, clock_mhz=300, pairs_per_event=1326, save_to=None):
    """results {variant name: read_csynth_report(...)}: resources (% of the U55C) and pairs per clock."""
    variants = list(results)
    colors = [CATEGORICAL[number] for number in range(len(variants))]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    resources = list(U55C_RESOURCES)
    positions = np.arange(len(resources))
    width = 0.8 / len(variants)
    for number, (variant, color) in enumerate(zip(variants, colors)):
        shares = [100 * results[variant][resource] / U55C_RESOURCES[resource] for resource in resources]
        bars = left.bar(positions + (number - (len(variants) - 1) / 2) * width, shares, width=width * 0.92,
                        color=color, label=variant)
        for bar, share in zip(bars, shares):
            left.text(bar.get_x() + bar.get_width() / 2, share + 0.4, f"{share:.1f}%", ha="center", va="bottom",
                      fontsize=7.5, color=INK_SECONDARY)
    left.set_xticks(positions)
    left.set_xticklabels(resources)
    left.set_ylabel("% of the U55C")
    left.yaxis.grid(True)
    left.set_axisbelow(True)
    left.legend(fontsize=8.5)
    left.set_title("Resources (Vitis HLS estimate)")

    event_us = [results[variant]["interval"] * pairs_per_event / clock_mhz for variant in variants]
    bars = right.barh(np.arange(len(variants)), event_us, color=colors, height=0.55)
    for bar, variant, value in zip(bars, variants, event_us):
        interval = results[variant]["interval"]
        every = "a new pair every clock" if interval == 1 else f"a new pair every {interval} clocks"
        right.text(value * 1.02, bar.get_y() + bar.get_height() / 2, f"{value:.1f} µs  ({every})", va="center",
                   fontsize=8.5, color=INK_SECONDARY)
    right.set_yticks(np.arange(len(variants)))
    right.set_yticklabels(variants)
    right.invert_yaxis()
    right.set_xlim(0, max(event_us) * 2.1)
    right.set_xlabel(f"µs for the {pairs_per_event:,} pairs of a median event, at {clock_mhz} MHz")
    right.xaxis.grid(True)
    right.set_axisbelow(True)
    right.set_title("Time for a median event")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_data_per_event(save_to=None):
    n_hits = np.arange(2, 141)
    pair_kilobytes = n_hits * (n_hits + 1) / 2 * 64 * 4 / 1000          # 4 bytes per float
    embedding_kilobytes = n_hits * 32 * 4 / 1000
    figure, axes = plt.subplots(figsize=(8.5, 3.6))
    axes.plot(n_hits, pair_kilobytes, color=CATEGORICAL[1], linewidth=2)
    axes.plot(n_hits, embedding_kilobytes, color=CATEGORICAL[0], linewidth=2)
    axes.text(n_hits[-1] + 2, pair_kilobytes[-1], "pair features [K, 64]\n(pairs built on the CPU)", va="center",
              fontsize=8.5, color=INK_SECONDARY)
    axes.text(n_hits[-1] + 2, embedding_kilobytes[-1], "embeddings [N, 32]\n(pairs built on the FPGA)", va="center",
              fontsize=8.5, color=INK_SECONDARY)
    axes.axvline(51, color=INK_MUTED, linestyle=":", linewidth=1.2)
    axes.text(53, pair_kilobytes[-1] * 0.5, f"median event, 51 hits:\n{51 * 52 / 2 * 256 / 1000:.0f} kB vs "
                                            f"{51 * 128 / 1000:.1f} kB", fontsize=8.5, color=INK)
    axes.set_yscale("log")
    axes.set_xlim(0, 175)
    axes.set_xlabel("hits per event, N")
    axes.set_ylabel("kB sent to the card per event")
    axes.yaxis.grid(True)
    axes.set_axisbelow(True)
    axes.set_title("Data sent to the FPGA for every event")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_kernel_pipeline(save_to=None):
    """The stages of the provided kernel, and what each one takes from hls4ml."""
    stages = [("LoadHits", "reads the N embeddings\nfrom the card memory", "the input type"),
              ("Hits", "first layer, per hit:\nU[i] and V[j]", "fc1 weights, products"),
              ("PairLoop", "one pair per clock:\nU[i] + V[j] → ReLU → fc2", "types, fc2 weights,\nproducts, roundings"),
              ("Candidate filter", "keeps the pairs that\ncan pass the threshold", "the logit type"),
              ("WriteCand", "writes (i, j, l0, l1)\nto the card memory", "")]
    figure, axes = plt.subplots(figsize=(12.5, 3.3))
    box_width, gap = 2.15, 0.45
    hls4ml_color = "#1f9d55"
    for number, (title, detail, from_hls4ml) in enumerate(stages):
        x = number * (box_width + gap)
        _box(axes, x, 0.9, box_width, 1.25, tint(FPGA_COLOR, 0.15), edge=FPGA_COLOR, line_width=1.6, rounding=0.1)
        axes.text(x + box_width / 2, 1.85, title, ha="center", va="center", fontsize=10, fontweight="bold")
        axes.text(x + box_width / 2, 1.33, detail, ha="center", va="center", fontsize=8.3, color=INK_SECONDARY)
        if from_hls4ml:
            _box(axes, x + 0.1, -0.05, box_width - 0.2, 0.72, tint(hls4ml_color, 0.18), edge=hls4ml_color,
                 line_width=1.2, rounding=0.08)
            axes.text(x + box_width / 2, 0.31, f"from hls4ml:\n{from_hls4ml}", ha="center", va="center",
                      fontsize=7.8, color=INK)
            axes.add_patch(FancyArrowPatch((x + box_width / 2, 0.68), (x + box_width / 2, 0.88), arrowstyle="-|>",
                                           mutation_scale=9, color=hls4ml_color, linewidth=1))
        if number < len(stages) - 1:
            axes.add_patch(FancyArrowPatch((x + box_width + 0.04, 1.52), (x + box_width + gap - 0.04, 1.52),
                                           arrowstyle="-|>", mutation_scale=11, color=INK_MUTED, linewidth=1.2))
    axes.set_xlim(-0.1, len(stages) * (box_width + gap) - gap + 0.1)
    axes.set_ylim(-0.1, 2.3)
    axes.axis("off")
    axes.legend(handles=[Patch(facecolor=tint(FPGA_COLOR, 0.15), edgecolor=FPGA_COLOR, label="custom C++ kernel (provided)"),
                         Patch(facecolor=tint(hls4ml_color, 0.18), edgecolor=hls4ml_color, label="generated by hls4ml")],
                loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2, fontsize=8.5)
    axes.set_title("The kernel on the FPGA: custom C++ around the hls4ml network", pad=14)
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_estimate_vs_routed(estimate, routed, save_to=None):
    """Vitis HLS estimate vs the resources after place and route, one small panel per resource."""
    resources = [("DSP", "DSP"), ("LUT", "LUT"), ("FF", "flip-flops"), ("BRAM_18K", "BRAM (18 Kb)"), ("URAM", "URAM")]
    figure, axes_list = plt.subplots(1, len(resources), figsize=(12.5, 3.2))
    for axes, (resource, label) in zip(axes_list, resources):
        values = [estimate[resource], routed[resource]]
        bars = axes.bar([0, 1], values, width=0.6, color=["#b5b3ab", CATEGORICAL[0]])
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:,}", ha="center", va="bottom",
                      fontsize=8.5, color=INK)
        axes.set_xticks([0, 1])
        axes.set_xticklabels(["HLS\nestimate", "after place\n& route"], fontsize=8.5)
        axes.set_ylim(0, max(values) * 1.22)
        axes.set_yticks([])
        axes.spines["left"].set_visible(False)
        axes.set_title(label, fontsize=10)
    figure.suptitle("The kernel: Vitis HLS estimate vs what the real bitstream uses", x=0.01, ha="left",
                    fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()



# ---------------------------------------------------------------------------------------------------------
# Exercise 3 · inference on the FPGA
# ---------------------------------------------------------------------------------------------------------
import subprocess

PATH_COLOR = {"full": CPU_COLOR, "split": "#8f8d86", "card": FPGA_COLOR}
PATH_LABEL = {"full": "CPU: full model", "split": "CPU: encoder + pair classifier", "card": "hybrid: CPU + U55C"}


def check_card_is_free(card_address):
    """Print what is loaded on the card and stop if one of its compute units is busy (the card is shared)."""
    status = subprocess.run(["xbutil", "examine", "-d", card_address, "-r", "dynamic-regions"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    for line in status.splitlines():
        if re.search(r"Xclbin UUID|pairedges|krnl|\(IDLE\)|\(BUSY\)|\(RUNNING\)", line):
            print(line.strip())
    if re.search(r"\((BUSY|RUNNING)\)", status):
        raise RuntimeError("a compute unit on the card is busy: someone else is using it, wait and try again")
    print("The card is free")


def plot_agreement(differing_p1, threshold, tracks, save_to=None):
    """differing_p1: the full model's p1 of the pairs that are an edge in one path only; tracks {path: [tp, fp, fn]}."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    half_width = max(5e-4, np.abs(differing_p1 - threshold).max() * 1.4) if len(differing_p1) else 2e-3
    left.hist(differing_p1, bins=np.linspace(threshold - half_width, threshold + half_width, 41), color=FPGA_COLOR,
              edgecolor=SURFACE, linewidth=1)
    left.axvline(threshold, color=CRITICAL, linestyle="--", linewidth=1.3)
    left.set_xlim(threshold - half_width, threshold + half_width)
    left.set_xlabel("p₁ of the full model (float)")
    left.set_ylabel("pairs")
    left.yaxis.grid(True)
    left.set_axisbelow(True)
    left.set_title(f"The {len(differing_p1)} edges that differ: all at the threshold")
    left.text(0.02, 0.95, f"threshold {threshold}\nwidest distance {np.abs(differing_p1 - threshold).max():.1e}"
              if len(differing_p1) else "no edge differs", transform=left.transAxes, va="top", fontsize=8.8)

    kinds = [("true positives", 0), ("false positives", 1), ("missed tracks", 2)]
    paths = list(tracks)
    positions = np.arange(len(kinds))
    width = 0.8 / len(paths)
    for number, path in enumerate(paths):
        values = [tracks[path][index] for _, index in kinds]
        bars = right.bar(positions + (number - (len(paths) - 1) / 2) * width, values, width=width * 0.92,
                         color=PATH_COLOR[path], label=PATH_LABEL[path])
        for bar, value in zip(bars, values):
            right.text(bar.get_x() + bar.get_width() / 2, value + 40, f"{value:,}", ha="center", va="bottom",
                       fontsize=7.5, color=INK_SECONDARY, rotation=90)
    right.set_xticks(positions)
    right.set_xticklabels([name for name, _ in kinds])
    right.set_ylim(0, max(max(values) for values in tracks.values()) * 1.3)
    right.set_ylabel("tracks")
    right.yaxis.grid(True)
    right.set_axisbelow(True)
    right.legend(fontsize=8, loc="upper right")
    right.set_title("Tracks reconstructed from the edges of each path")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_latency(total_us, hits_per_event, save_to=None):
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    for path, times in total_us.items():
        ordered = np.sort(times)
        left.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=PATH_COLOR[path], linewidth=2,
                  label=f"{PATH_LABEL[path]} (median {np.median(times):.0f} µs)")
    left.set_xscale("log")
    left.set_xlabel("time per event (µs)")
    left.set_ylabel("fraction of the events")
    left.grid(True)
    left.set_axisbelow(True)
    left.legend(fontsize=8, loc="lower right")
    left.set_title("Distribution of the time per event")

    n_grid = np.linspace(hits_per_event.min(), hits_per_event.max(), 100)
    for path in ("full", "card"):
        right.scatter(hits_per_event, total_us[path], s=9, color=PATH_COLOR[path], alpha=0.45, edgecolors="none")
        fit = np.polyfit(hits_per_event, total_us[path], 2)
        right.plot(n_grid, np.polyval(fit, n_grid), color=PATH_COLOR[path], linewidth=2, label=PATH_LABEL[path])
    right.set_xlabel("hits per event, N")
    right.set_ylabel("time per event (µs)")
    right.grid(True)
    right.set_axisbelow(True)
    right.legend(fontsize=8, loc="upper left")
    right.set_title("Time vs event size (lines: quadratic fits)")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


STAGE_COLOR = {"model": CPU_COLOR, "encoder": tint(CPU_COLOR, 0.55), "pairs": "#b5b3ab", "head": CPU_COLOR,
               "card": FPGA_COLOR, "edges": "#d8d6cf"}
STAGE_LABEL = {"model": "full model", "encoder": "encoder", "pairs": "pairs", "head": "pair classifier",
               "card": "card call", "edges": "edges"}


def plot_stages(timings, total_us, stages, save_to=None):
    """timings {path: [{stage: µs}, ...]}: the mean time of each stage, one stacked bar per path."""
    figure, axes = plt.subplots(figsize=(11, 2.9))
    paths = list(stages)
    for row, path in enumerate(paths):
        left = 0
        for stage in stages[path]:
            width = float(np.mean([times[stage] for times in timings[path]]))
            axes.barh(row, width, left=left, height=0.6, color=STAGE_COLOR[stage], edgecolor=SURFACE, linewidth=2)
            if width > 22:
                dark = stage in ("model", "head", "card")
                axes.text(left + width / 2, row, f"{STAGE_LABEL[stage]}\n{width:.0f} µs", ha="center", va="center",
                          fontsize=7.8, color="white" if dark else INK)
            left += width
        mean_total = float(total_us[path].mean())
        axes.barh(row, mean_total - left, left=left, height=0.6, color=NEUTRAL_TILE, edgecolor=SURFACE, linewidth=2)
        axes.text(mean_total + 6, row, f"{mean_total:.0f} µs", va="center", fontsize=9, color=INK, fontweight="bold")
    axes.set_yticks(range(len(paths)))
    axes.set_yticklabels([PATH_LABEL[path] for path in paths])
    axes.invert_yaxis()
    axes.set_xlim(0, max(float(times.mean()) for times in total_us.values()) * 1.12)
    axes.set_xlabel("mean µs per event (light grey: Python between the stages)")
    axes.xaxis.grid(True)
    axes.set_axisbelow(True)
    axes.tick_params(axis="y", length=0)
    axes.set_title("Where the time goes")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_card_call(card_timings, pairs_per_event, kernel_mhz, save_to=None):
    """card_timings: the driver's timers of each event ({"card_h2d_us", "card_kernel_us", "card_d2h_us", ...})."""
    parts = [("card_h2d_us", "to the card (PCIe)", CATEGORICAL[0]), ("card_kernel_us", "kernel", CATEGORICAL[1]),
             ("card_d2h_us", "back from the card (PCIe)", CATEGORICAL[2])]
    figure, axes = plt.subplots(figsize=(9, 3.8))
    for key, label, color in parts:
        values = [times[key] for times in card_timings]
        axes.scatter(pairs_per_event, values, s=10, color=color, alpha=0.6, edgecolors="none",
                     label=f"{label}: median {np.median(values):.1f} µs")
    order = np.argsort(pairs_per_event)
    axes.plot(pairs_per_event[order], pairs_per_event[order] / kernel_mhz, color=INK_MUTED, linewidth=1.2,
              linestyle="--", label=f"one pair per clock at {kernel_mhz:.0f} MHz")
    axes.set_xlabel("pairs per event, K")
    axes.set_ylabel("µs")
    axes.grid(True)
    axes.set_axisbelow(True)
    axes.legend(fontsize=8, loc="upper left")
    axes.set_title("Anatomy of one card call")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_batching(batched_us, full_median, encoder_median, save_to=None):
    sizes = list(batched_us)
    values = [batched_us[size] for size in sizes]
    figure, axes = plt.subplots(figsize=(9, 3.8))
    axes.plot(sizes, values, "o-", color=FPGA_COLOR, linewidth=2, markersize=7, markeredgecolor=SURFACE,
              markeredgewidth=1.5)
    for size, value in zip(sizes, values):
        axes.text(size, value + 6, f"{value:.0f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
    axes.axhline(full_median, color=CPU_COLOR, linestyle="--", linewidth=1.5)
    axes.text(sizes[-1], full_median + 6, "CPU: full model, 1 thread", ha="right", va="bottom", fontsize=8.5,
              color=INK_SECONDARY)
    axes.axhline(encoder_median, color=INK_MUTED, linestyle=":", linewidth=1.5)
    axes.text(sizes[-1], encoder_median - 6, "the encoder alone (CPU)", ha="right", va="top", fontsize=8.5,
              color=INK_SECONDARY)
    axes.set_xscale("log", base=2)
    axes.set_xticks(sizes)
    axes.set_xticklabels([str(size) for size in sizes])
    axes.set_ylim(0, max(values + [full_median]) * 1.18)
    axes.set_xlabel("events per card call")
    axes.set_ylabel("µs per event (median)")
    axes.yaxis.grid(True)
    axes.set_axisbelow(True)
    axes.set_title("Batching: the fixed cost of a card call is shared by the events of the batch")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_threads(cpu_us_by_threads, hybrid_us_by_threads, save_to=None):
    counts = list(cpu_us_by_threads)
    positions = np.arange(len(counts))
    figure, axes = plt.subplots(figsize=(8, 3.4))
    for offset, values, path in ((-0.19, cpu_us_by_threads, "full"), (0.19, hybrid_us_by_threads, "card")):
        bars = axes.bar(positions + offset, [values[count] for count in counts], width=0.36, color=PATH_COLOR[path],
                        label=PATH_LABEL[path])
        for bar, count in zip(bars, counts):
            axes.text(bar.get_x() + bar.get_width() / 2, values[count] + 4, f"{values[count]:.0f}", ha="center",
                      va="bottom", fontsize=8, color=INK_SECONDARY)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{count} thread{'s' if count > 1 else ''}" for count in counts])
    axes.set_ylabel("µs per event (median)")
    axes.set_ylim(0, max(cpu_us_by_threads.values()) * 1.2)
    axes.yaxis.grid(True)
    axes.set_axisbelow(True)
    axes.legend(fontsize=8.5)
    axes.set_title("ONNX Runtime threads on the CPU (the hybrid: threads of its encoder)")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()


def plot_summary(results, save_to=None):
    """results: [(label, µs per event, path)], the first one is the reference for the speed-ups."""
    figure, axes = plt.subplots(figsize=(10, 0.55 * len(results) + 1.2))
    reference = results[0][1]
    for row, (label, value, path) in enumerate(results):
        axes.barh(row, value, height=0.6, color=PATH_COLOR[path])
        axes.text(value + 4, row, f"{value:.0f} µs   ×{reference / value:.2f}", va="center", fontsize=9, color=INK)
    axes.set_yticks(range(len(results)))
    axes.set_yticklabels([label for label, _, _ in results])
    axes.invert_yaxis()
    axes.set_xlim(0, max(value for _, value, _ in results) * 1.3)
    axes.set_xlabel(f"µs per event (median); ×: speed compared with \"{results[0][0]}\"")
    axes.xaxis.grid(True)
    axes.set_axisbelow(True)
    axes.tick_params(axis="y", length=0)
    axes.set_title("The GTK Transformer: CPU vs hybrid inference with the Alveo U55C")
    plt.tight_layout()
    _save(figure, save_to)
    plt.show()
