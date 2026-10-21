"""Charts and self-checks for the school notebook 00 (hls4ml on the full model).

The notebook imports this module so that it can stay focused on hls4ml: nothing here is needed to solve the
exercises. Each check_exercise_N function prints one line: ⏳ not done yet, ✅ correct, or ❌ with a hint.
"""
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import hls4ml

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
        print("⏳ Complete Exercise 1 first: this chart shows the two lists you get there.")
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
        print("⏳ Complete Exercises 1 and 3 first: this chart uses onnx_layers and operation_counts.")
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
        print("⏳ Complete Exercise 1 first: this chart uses onnx_layers.")
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


def load_hits_per_event(config, model_config):
    """Number of hits of each test event, after the repo's clustering (as the model sees them)."""
    import hi_common
    from hybrid import cluster_event
    tracking = hi_common.import_metrics(config)
    hits = []
    for event in hi_common.load_events(config, model_config, config["data"]["eval_split"]):
        _, clustered_hits, _ = cluster_event(tracking, event["df"], event["features"])
        hits.append(len(clustered_hits))
    return np.array(hits)


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
    parts = [("Encoder", "embedding +\n2 Transformer layers", False, CPU_COLOR, "CPU · ONNX Runtime"),
             ("Pair building", "picks the embeddings\nof hits i and j", False, FPGA_COLOR, "FPGA · custom C++ kernel"),
             ("Pair classifier", "Linear → ReLU → Linear", True, FPGA_COLOR, "FPGA · hls4ml")]
    figure, axes = plt.subplots(figsize=(11, 3.0))
    box_width, gap = 3.0, 0.7
    for number, (title, detail, convertible, color, destination) in enumerate(parts):
        x = number * (box_width + gap)
        _box(axes, x, 0.55, box_width, 1.35, tint(color, 0.15), edge=color, line_width=1.8, rounding=0.12)
        axes.text(x + box_width / 2, 1.58, title, ha="center", va="center", fontsize=11, fontweight="bold")
        axes.text(x + box_width / 2, 1.08, detail, ha="center", va="center", fontsize=8.8, color=INK_SECONDARY)
        icon, icon_color = ("✓", GOOD) if convertible else ("✗", CRITICAL)
        axes.text(x + 0.18, 1.72, icon, fontsize=13, color=icon_color, fontweight="bold", va="center")
        axes.text(x + box_width / 2, 0.68, "hls4ml can convert it" if convertible else "hls4ml cannot convert it",
                  ha="center", va="center", fontsize=8, color=INK_SECONDARY, style="italic")
        axes.text(x + box_width / 2, 0.2, f"→ {destination}", ha="center", va="center", fontsize=9.5,
                  fontweight="bold", color=INK)
        if number < len(parts) - 1:
            axes.add_patch(FancyArrowPatch((x + box_width + 0.08, 1.22), (x + box_width + gap - 0.08, 1.22),
                                           arrowstyle="-|>", mutation_scale=14, color=INK_MUTED, linewidth=1.3))
    axes.set_xlim(-0.1, len(parts) * (box_width + gap) - gap + 0.1)
    axes.set_ylim(0, 2.1)
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
        print("⏳ Complete the conversion first.")
        return
    print(f"{'layer':8s} {'type':11s} fixed-point types")
    for layer in hls_model.get_layers():
        types = ", ".join(f"{kind.replace('_t', '')}: {variable.precision}" for kind, variable in layer.types.items())
        print(f"{layer.name:8s} {layer.class_name:11s} {types}")


def first_unsupported_operation(onnx_model):
    supported = set(hls4ml.converters.get_supported_onnx_layers()) | set(SKIPPED_BY_HLS4ML_ONNX)
    for node in onnx_model.graph.node:
        if node.op_type not in supported:
            return node.op_type
    return None


# ---------------------------------------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------------------------------------
RESULTS = {}
EXERCISE_TITLES = {1: "What can hls4ml read?", 2: "Is the MLP supported?", 3: "Which operations does it use?",
                   4: "Ask hls4ml (ONNX)", 5: "The PyTorch frontend", 6: "Check the pair classifier"}


def _report(number, status, message):
    icon = {None: "⏳", True: "✅", False: "❌"}[status]
    RESULTS[number] = status
    print(f"{icon} Exercise {number}: {message}")


def _as_sorted_list(values):
    try:
        return sorted(values, key=str)
    except TypeError:
        return None


def check_exercise_1(onnx_layers, pytorch_layers):
    if onnx_layers is None or pytorch_layers is None:
        return _report(1, None, "not done yet: replace the two None values")
    expected_onnx = sorted(hls4ml.converters.get_supported_onnx_layers(), key=str)
    expected_pytorch = sorted(hls4ml.converters.get_supported_pytorch_layers(), key=str)
    if _as_sorted_list(onnx_layers) == expected_pytorch and _as_sorted_list(pytorch_layers) == expected_onnx:
        return _report(1, False, "the two lists are swapped")
    if _as_sorted_list(onnx_layers) != expected_onnx:
        return _report(1, False, "onnx_layers is not the ONNX list: look for get_supported_..._layers in hls4ml.converters")
    if _as_sorted_list(pytorch_layers) != expected_pytorch:
        return _report(1, False, "pytorch_layers is not the PyTorch list: look for get_supported_..._layers in hls4ml.converters")
    _report(1, True, f"the ONNX frontend reads {len(expected_onnx)} operations, the PyTorch frontend {len(expected_pytorch)} layers")


def _is_hls_model(value):
    return value is not None and hasattr(value, "get_layers")


def check_exercise_2(mlp_unsupported, mlp_hls_model):
    if mlp_unsupported is None:
        return _report(2, None, "not done yet: part a) fills mlp_unsupported")
    found = list(mlp_unsupported)
    if found:
        if "SmallMLP" in found:
            return _report(2, False, "SmallMLP is the model itself (the first item of named_modules(), with an empty name): skip it")
        return _report(2, False, f"{found} are in pytorch_layers: compare type(module).__name__ with the list")
    if mlp_hls_model is None:
        return _report(2, None, "part a) is right (every module is supported); part b) is not done yet")
    if not _is_hls_model(mlp_hls_model):
        return _report(2, False, "mlp_hls_model should be what convert_from_pytorch_model returns")
    layer_names = [layer.name for layer in mlp_hls_model.get_layers()]
    if not {"fc1", "relu", "fc2"} <= set(layer_names):
        return _report(2, False, f"the hls4ml model has layers {layer_names}: did you convert mlp?")
    _report(2, True, f"every module is supported, and hls4ml built the model: {' → '.join(layer_names)}")


def check_exercise_3(operation_counts, unsupported_operations, model):
    if operation_counts is None or unsupported_operations is None:
        return _report(3, None, "not done yet: fill operation_counts and unsupported_operations")
    expected_counts = Counter(node.op_type for node in model.graph.node)
    supported = set(hls4ml.converters.get_supported_onnx_layers()) | set(SKIPPED_BY_HLS4ML_ONNX)
    expected_unsupported = {operation for operation in expected_counts if operation not in supported}
    try:
        counts = {name: int(count) for name, count in dict(operation_counts).items()}
    except (TypeError, ValueError):
        return _report(3, False, "operation_counts should map each operation type to its number of nodes")
    if counts != dict(expected_counts):
        if sum(counts.values()) != sum(expected_counts.values()):
            return _report(3, False, f"your counts add up to {sum(counts.values())} nodes, the graph has "
                                     f"{sum(expected_counts.values())}: count every node of model.graph.node")
        return _report(3, False, "the total is right but some counts are not: count node.op_type")
    if set(unsupported_operations) != expected_unsupported:
        missing = expected_unsupported - set(unsupported_operations)
        extra = set(unsupported_operations) - expected_unsupported
        if extra:
            return _report(3, False, f"{sorted(extra)} are supported (or skipped by hls4ml): check them against onnx_layers")
        return _report(3, False, f"{len(missing)} unsupported operation type(s) missing")
    n_nodes = sum(expected_counts[operation] for operation in expected_unsupported)
    _report(3, True, f"{len(expected_unsupported)} of the {len(expected_counts)} operation types are not supported "
                     f"({n_nodes} of {sum(expected_counts.values())} nodes)")


def check_exercise_4(onnx_error, folded_error, model, folded_model):
    if onnx_error is None:
        return _report(4, None, "not done yet: call the ONNX converter inside the try block")
    expected_first = first_unsupported_operation(model)
    if expected_first not in str(onnx_error):
        return _report(4, False, f"unexpected message {onnx_error!r}: call convert_from_onnx_model on model")
    if folded_error is None:
        return _report(4, None, f"part a) is right (hls4ml stops at {expected_first}); part b) is not done yet")
    expected_next = first_unsupported_operation(folded_model)
    if expected_next not in str(folded_error):
        return _report(4, False, f"unexpected message {folded_error!r}: convert folded_model this time")
    _report(4, True, f"hls4ml stops at {expected_first}, and after folding the constants at {expected_next}")


def check_exercise_5(unsupported_modules, pytorch_error, trained_model):
    if unsupported_modules is None:
        return _report(5, None, "not done yet: fill unsupported_modules")
    supported = set(hls4ml.converters.get_supported_pytorch_layers())
    expected = set()
    for name, module in trained_model.named_modules():
        class_name = type(module).__name__
        if name and class_name not in supported | {"Dropout", "Sequential", "ModuleList"}:
            expected.add(class_name)
    found = set(unsupported_modules)
    if "TrainedTransformer" in found:
        return _report(5, False, "TrainedTransformer is the model itself: skip the item with an empty name")
    if found & {"ModuleList", "Dropout"}:
        return _report(5, False, "ModuleList (a container) and Dropout (removed by hls4ml) do not count")
    if found != expected:
        missing = expected - found
        if missing:
            return _report(5, False, f"{len(missing)} class name(s) missing: look at every level, inside the attention too")
        return _report(5, False, f"{sorted(found - expected)} are supported")
    if pytorch_error is None:
        return _report(5, None, "part a) is right; part b) is not done yet")
    if "TransformerEncoder" not in str(pytorch_error):
        return _report(5, False, f"unexpected message {pytorch_error!r}: call config_from_pytorch_model on trained_model")
    _report(5, True, f"not supported: {', '.join(sorted(expected))}; hls4ml says: {pytorch_error}")


def check_exercise_6(classifier_hls_model):
    if classifier_hls_model is None:
        return _report(6, None, "not done yet")
    if not _is_hls_model(classifier_hls_model):
        return _report(6, False, "classifier_hls_model should be what convert_from_pytorch_model returns")
    layer_names = [layer.name for layer in classifier_hls_model.get_layers()]
    if not {"fc1", "relu", "fc2"} <= set(layer_names):
        return _report(6, False, f"the hls4ml model has layers {layer_names}: did you convert trained_classifier?")
    _report(6, True, f"hls4ml converts the trained pair classifier: {' → '.join(layer_names)}")


def show_scoreboard():
    print("Your scoreboard")
    for number, title in EXERCISE_TITLES.items():
        icon = {None: "⏳", True: "✅", False: "❌"}[RESULTS.get(number)]
        print(f"  {icon}  {number}. {title}")
    n_done = sum(1 for status in RESULTS.values() if status)
    print(f"\n{n_done} / {len(EXERCISE_TITLES)} exercises done"
          + ("  🎉 ready for notebook 1!" if n_done == len(EXERCISE_TITLES) else ""))
