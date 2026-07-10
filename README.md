# POSITMJ

*by Mihir and Jiyan*

[![PyPI version](https://img.shields.io/pypi/v/POSITMJ.svg)](https://pypi.org/project/POSITMJ/)
[![Python versions](https://img.shields.io/pypi/pyversions/POSITMJ.svg)](https://pypi.org/project/POSITMJ/)
[![License](https://img.shields.io/pypi/l/POSITMJ.svg)](https://pypi.org/project/POSITMJ/)
[![Downloads](https://img.shields.io/pypi/dm/POSITMJ.svg)](https://pypi.org/project/POSITMJ/)

**POSITMJ** is a Python library for building, training, and exporting Multi-Layer Perceptron (MLP) neural networks with **Posit number format** support, designed for FPGA/ASIC hardware deployment.

It takes you from a raw dataset all the way to synthesizable Verilog, in four stages:

1. **Load a dataset** (built-in or your own CSV)
2. **Choose an MLP architecture and train it** (e.g. N-16-8-3)
3. **Convert the trained weights/biases to Posit<n, es> format**
4. **Export as a Verilog LUT (ROM) module or a `.mem` hex file**

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Step-by-Step Usage](#step-by-step-usage)
  - [1. Load a Dataset](#1-load-a-dataset)
  - [2. Build and Train the MLP](#2-build-and-train-the-mlp)
  - [3. Convert to Posit Format](#3-convert-to-posit-format)
  - [4. Export to Hardware (LUT or .mem)](#4-export-to-hardware-lut-or-mem)
- [Using Your Own External Dataset](#using-your-own-external-dataset)
- [Custom Architectures (e.g. N-16-8-3)](#custom-architectures-eg-n-16-8-3)
- [API Reference](#api-reference)
  - [DatasetLoader](#datasetloader)
  - [MLPBuilder](#mlpbuilder)
  - [PositConverter](#positconverter)
  - [PositExporter](#positexporter)
  - [PositMLPPipeline](#positmlppipeline)
- [Output File Formats](#output-file-formats)
- [Testing & Validation](#testing--validation)
- [Requirements](#requirements)
- [Known Limitations](#known-limitations)
- [Links](#links)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Load built-in datasets (`iris`, `wine`, `breast_cancer`) or any external `.csv` file
- Automatic train/test split and feature normalization (`StandardScaler`)
- Configurable MLP architecture (any number/size of hidden layers) for classification or regression
- Input layer size (N) and output layer size are inferred automatically from your data
- Custom Posit encoder supporting arbitrary bit-width `n` and exponent size `es` (e.g. Posit16, es=1)
- Converts all trained weights and biases, layer by layer, to posit bit patterns
- Export as:
  - A synthesizable **Verilog LUT/ROM module** (`.v`)
  - A plain **hex `.mem` file** (for `$readmemh` in testbenches)
- Interactive export mode that asks the user whether they want a LUT or a `.mem` file
- Single end-to-end orchestrator class (`PositMLPPipeline`) or use each stage independently

---

## Installation

```bash
pip install POSITMJ
```

That's it — POSITMJ is published on PyPI: https://pypi.org/project/POSITMJ/

For local/editable development (from source):

```bash
git clone https://github.com/yourusername/POSITMJ.git
cd POSITMJ
pip install -e .
```

### Dependencies

POSITMJ requires:

```
numpy
pandas
scikit-learn
```

These are installed automatically with `pip install POSITMJ`. Install manually only if needed:

```bash
pip install numpy pandas scikit-learn
```

---

## Project Structure

```
POSITMJ/
├── POSITMJ/
│   └── __init__.py       # Core library (all 4 stages)
├── tests/
│   └── test_positmj.py   # pytest suite (35 tests)
├── pyproject.toml         # Package metadata (name = "POSITMJ")
├── LICENSE                # MIT
├── CHANGELOG.md
├── README.md              # This file
```

---

## Quick Start

```python
from POSITMJ import PositMLPPipeline

# Create the pipeline: choose posit format (n bits, es exponent size)
pipeline = PositMLPPipeline(n=16, es=1)

# Stage 1: Load a dataset (built-in example here)
pipeline.load_dataset("iris")

# Stage 2: Choose architecture and train
pipeline.build_and_train(hidden_layers=(8, 8), task="classification")

# Stage 3: Convert trained weights/biases to Posit format
pipeline.convert_to_posit()

# Stage 4: Export to hardware format
pipeline.export(mode="lut")     # or mode="mem"
```

This trains a 4-8-8-3 MLP on Iris, converts every weight/bias to Posit16(es=1), and writes Verilog LUT modules to `posit_export/`.

---

## Step-by-Step Usage

### 1. Load a Dataset

**Built-in dataset:**

```python
pipeline.load_dataset("iris")           # or "wine", "breast_cancer"
```

**External CSV dataset:**

```python
pipeline.load_dataset(
    source="/path/to/your_dataset.csv",
    target_column="label",   # name of the column containing class labels
    normalize=True,           # StandardScaler normalization (recommended)
    test_size=0.2             # 20% held out for testing
)
```

Your CSV should look like this — feature columns plus one label column:

```
feat1,feat2,feat3,...,featN,label
0.12,0.45,0.98,...,        , 0
```

### 2. Build and Train the MLP

```python
pipeline.build_and_train(
    hidden_layers=(16, 8),     # any number/size of hidden layers
    task="classification",     # or "regression"
    activation="relu",         # "relu", "tanh", "logistic", etc.
    max_iter=3000
)
```

- **Input size (N)** is inferred automatically from the number of feature columns.
- **Output size** is inferred automatically from the number of classes (classification) or set to 1 (regression).
- Training accuracy/score is printed automatically after training.

### 3. Convert to Posit Format

```python
pipeline.convert_to_posit(n=16, es=1)
```

- `n`: total posit bit-width (e.g. 8, 16, 32)
- `es`: exponent size (e.g. 0, 1, 2)

If you already set `n`/`es` when creating `PositMLPPipeline(...)`, you can call `convert_to_posit()` with no arguments.

### 4. Export to Hardware (LUT or .mem)

**Programmatic (you choose the format in code):**

```python
pipeline.export(mode="lut")   # Verilog LUT/ROM module (.v)
pipeline.export(mode="mem")   # Hex .mem file (.mem)
```

**Interactive (asks the user at runtime):**

```python
pipeline.export_interactive()
```

This will prompt:

```
Export format - type 'lut' for a Verilog LUT module, or 'mem' for a .mem hex file:
```

Output files are written to the `output_dir` you specified (default: `posit_export/`), one file per weight matrix and bias vector, e.g.:

```
posit_export/
├── layer0_weights.v
├── layer0_biases.v
├── layer1_weights.v
├── layer1_biases.v
```

---

## Using Your Own External Dataset

```python
from POSITMJ import PositMLPPipeline

pipeline = PositMLPPipeline(n=16, es=1, output_dir="my_export")

pipeline.load_dataset(
    source="/home/user/data/sensor_data.csv",
    target_column="class",
    normalize=True,
    test_size=0.2
)

pipeline.build_and_train(hidden_layers=(16, 8), task="classification")
pipeline.convert_to_posit()
pipeline.export_interactive()
```

**Notes:**
- String class labels (e.g. `"setosa"`, `"versicolor"`) work fine — no manual encoding needed.
- If a class has very few samples, training may still work, but consider collecting more data per class for stable accuracy.

---

## Custom Architectures (e.g. N-16-8-3)

The hidden layer sizes are fully configurable, and N (input) / output size are inferred from the data — so an **N-16-8-3** network is simply:

```python
pipeline.build_and_train(
    hidden_layers=(16, 8),   # the "16-8" part
    task="classification"     # output size auto-inferred (3 in this example)
)
```

If your dataset has, say, 11 features and 3 classes, this automatically becomes an **11-16-8-3** network — no extra configuration required.

---

## API Reference

### `DatasetLoader`

```python
DatasetLoader(normalize=True, test_size=0.2, random_state=42)
```

| Method | Description |
|---|---|
| `.load(source, target_column=None)` | Loads a built-in dataset name or CSV path. Returns `X_train, X_test, y_train, y_test`. |

---

### `MLPBuilder`

```python
MLPBuilder(hidden_layers=(8, 8), task="classification", activation="relu", max_iter=2000, random_state=42)
```

| Method | Description |
|---|---|
| `.train(X_train, y_train)` | Trains the MLP. |
| `.evaluate(X_test, y_test)` | Returns accuracy (classification) or R² score (regression). |
| `.get_weights_and_biases()` | Returns `(weights_list, biases_list)` from the trained model. |

---

### `PositConverter`

```python
PositConverter(n=16, es=1)
```

| Method | Description |
|---|---|
| `.float_to_posit_bits(value)` | Converts a single float to an n-bit posit integer. |
| `.convert_array(arr)` | Converts a numpy array of floats to posit bit patterns. |
| `.convert_weights(weights_list, biases_list)` | Converts a full set of MLP weights/biases. |

---

### `PositExporter`

```python
PositExporter(n=16, output_dir="posit_export")
```

| Method | Description |
|---|---|
| `.to_mem_file(posit_array, filename)` | Writes a `.mem` hex file. |
| `.to_lut_verilog(posit_array, module_name, filename=None)` | Writes a Verilog LUT/ROM module. |
| `.export(posit_array, name, mode="lut")` | Dispatches to either export method based on `mode`. |

---

### `PositMLPPipeline`

```python
PositMLPPipeline(n=16, es=1, output_dir="posit_export")
```

| Method | Description |
|---|---|
| `.load_dataset(source, target_column=None, normalize=True, test_size=0.2, random_state=42)` | Stage 1 |
| `.build_and_train(hidden_layers=(8,8), task="classification", activation="relu", max_iter=2000, random_state=42)` | Stage 2 |
| `.convert_to_posit(n=None, es=None)` | Stage 3 |
| `.export(mode="lut")` | Stage 4 (programmatic) |
| `.export_interactive()` | Stage 4 (prompts user for `lut`/`mem`) |

---

## Output File Formats

### `.mem` file

Plain hex, one value per line, zero-padded to the posit bit-width (in hex digits):

```
0041
3f9a
0000
...
```

Designed for use with Verilog's `$readmemh` in testbenches or ROM initialization.

### `.v` LUT module

A synthesizable combinational ROM, e.g.:

```verilog
module layer0_weights (
    input  wire [3:0] addr,
    output reg  [15:0] data
);

always @(*) begin
    case (addr)
        4'd0: data = 16'd65;
        4'd1: data = 16'd16282;
        ...
        default: data = 16'd0;
    endcase
end

endmodule
```

---

## Testing & Validation

POSITMJ ships with a pytest suite (`tests/test_positmj.py`) covering:

- Exhaustive round-trip encode/decode correctness for small posit widths, and
  large random samples for wider ones
- Zero, NaN/Inf, and extreme-magnitude edge cases (posits never round a
  nonzero value down to exact zero)
- Round-half-to-even tie-breaking at exact regime/exponent boundaries
- Input validation across `DatasetLoader`, `MLPBuilder`, and `PositExporter`
- A full end-to-end pipeline test (dataset -> train -> convert -> export)

The posit encoder was additionally fuzz-tested against **SoftPosit**
(Berkeley's reference C posit implementation) across tens of thousands of
random values plus every exact power-of-two boundary, for all three standard
posit configurations (posit8/es=0, posit16/es=1, posit32/es=2), with **zero
bit-pattern mismatches**. That cross-check is included as an optional test
(`TestAgainstSoftPositReference`) that runs automatically if `softposit` is
installed, and is skipped otherwise:

```bash
pip install pytest softposit --break-system-packages
pytest tests/ -v
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn

---

## Known Limitations

- Non-standard `(n, es)` combinations that don't correspond to a widely-used
  reference implementation haven't been cross-validated against external
  libraries (only self-consistency round-trip tested) - the standard
  configurations (posit8/es=0, posit16/es=1, posit32/es=2) are the most
  thoroughly verified.
- Training uses scikit-learn's `MLPClassifier`/`MLPRegressor` (software
  training only); POSITMJ does not train on hardware - it trains in
  software, then converts and exports the final weights for hardware
  inference.
- Very small or highly imbalanced datasets may cause instability during
  `train_test_split` (e.g. class imbalance) - consider adjusting `test_size`
  or supplying more samples per class.

---

## Links

- **PyPI:** https://pypi.org/project/POSITMJ/
- **Source & issue tracker:** https://github.com/yourusername/POSITMJ
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

## Contributing

Issues and pull requests are welcome — in particular, cross-validation of
additional `(n, es)` combinations against other posit reference
implementations, and a `predict()`-style software inference helper that runs
a forward pass using the posit-quantized weights would be great additions.

---

## License

MIT License - see [LICENSE](LICENSE) for details.
