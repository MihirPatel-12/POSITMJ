"""
POSITMJ
=======

A Python library for building, training, and exporting Multi-Layer Perceptron
(MLP) neural networks with Posit number format support, for FPGA/ASIC
hardware deployment.

Pipeline stages
---------------
    1. Dataset loading & preprocessing      -> DatasetLoader
    2. MLP architecture definition & train  -> MLPBuilder
    3. Conversion to Posit<n, es> format    -> PositConverter
    4. Hardware export (LUT / .mem file)    -> PositExporter

All four stages are wired together by ``PositMLPPipeline`` for a simple
end-to-end workflow.

Example
-------
    >>> from POSITMJ import PositMLPPipeline
    >>> pipe = PositMLPPipeline(n=16, es=1)
    >>> pipe.load_dataset("iris")
    >>> pipe.build_and_train(hidden_layers=(8, 8), task="classification")
    >>> pipe.convert_to_posit()
    >>> pipe.export(mode="lut")      # or mode="mem", or export_interactive()
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

__version__ = "0.1.0"
__all__ = [
    "DatasetLoader",
    "MLPBuilder",
    "PositConverter",
    "PositExporter",
    "PositMLPPipeline",
]


# ---------------------------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------------------------
class DatasetLoader:
    """Load and preprocess datasets for MLP training."""

    BUILTIN = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }

    def __init__(self, normalize=True, test_size=0.2, random_state=42):
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler() if normalize else None

    def load(self, source, target_column=None):
        """
        Parameters
        ----------
        source : str
            Either a built-in dataset name ('iris', 'wine', 'breast_cancer')
            or a path to a .csv file.
        target_column : str, optional
            Required when ``source`` is a CSV path - name of the label column.

        Returns
        -------
        X_train, X_test, y_train, y_test : np.ndarray
        """
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"source must be a non-empty string (a built-in dataset name "
                f"{list(self.BUILTIN)} or a path to a .csv file); got {source!r}."
            )

        if source.lower() in self.BUILTIN:
            data = self.BUILTIN[source.lower()]()
            X, y = data.data, data.target
        elif source.endswith(".csv"):
            if target_column is None:
                raise ValueError(
                    "target_column must be specified for CSV datasets "
                    "(name of the column containing the labels)."
                )
            if not os.path.exists(source):
                raise FileNotFoundError(f"CSV file not found: {source}")
            df = pd.read_csv(source)
            if target_column not in df.columns:
                raise ValueError(
                    f"target_column {target_column!r} not found in {source}. "
                    f"Available columns: {list(df.columns)}"
                )
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
            if X.shape[1] == 0:
                raise ValueError(
                    "No feature columns left after removing target_column - "
                    "check that the CSV has more than just the label column."
                )
        else:
            raise ValueError(
                f"Unrecognized dataset source: {source!r}. Use a built-in "
                f"name {list(self.BUILTIN)} or a path ending in '.csv'."
            )

        n_samples = X.shape[0]
        if n_samples < 4:
            raise ValueError(
                f"Dataset has only {n_samples} samples - need at least a "
                f"few to form a train/test split."
            )

        unique_classes, class_counts = np.unique(y, return_counts=True)
        looks_like_classification = len(unique_classes) <= max(20, n_samples // 5)
        stratify = None
        if looks_like_classification:
            if class_counts.min() >= 2:
                stratify = y
            else:
                print(
                    "[DatasetLoader] Warning: at least one class has fewer "
                    "than 2 samples, so the train/test split is not "
                    "stratified. Consider adding more samples for that class."
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify,
        )

        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. MLP Architecture & Training
# ---------------------------------------------------------------------------
class MLPBuilder:
    """Define and train an MLP for a chosen architecture."""

    def __init__(self, hidden_layers=(8, 8), task="classification",
                 activation="relu", max_iter=2000, random_state=42):
        """
        Parameters
        ----------
        hidden_layers : tuple
            Sizes of hidden layers, e.g. (8, 8) for a 4-8-8-3 network
            (input/output sizes are inferred automatically from the data).
        task : str
            'classification' or 'regression'.
        """
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}."
            )
        if not hidden_layers or any(h <= 0 for h in hidden_layers):
            raise ValueError(
                f"hidden_layers must be a non-empty tuple of positive "
                f"integers, got {hidden_layers!r}."
            )

        self.hidden_layers = hidden_layers
        self.task = task
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train):
        if len(X_train) == 0:
            raise ValueError("X_train is empty - nothing to train on.")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}."
            )
        if not np.all(np.isfinite(X_train)):
            raise ValueError(
                "X_train contains NaN or infinite values - clean the "
                "dataset before training."
            )

        if self.task == "classification":
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        elif self.task == "regression":
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.score(X_test, y_test)

    def get_weights_and_biases(self):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.coefs_, self.model.intercepts_


# ---------------------------------------------------------------------------
# 3. Posit Conversion
# ---------------------------------------------------------------------------
class PositConverter:
    """
    Converts IEEE-754 floating point values to Posit<n, es> bit patterns.
    Supports arbitrary total width n and exponent size es
    (e.g. Posit16, es=1 as used in your ICBHI / Iris MLP work).
    """

    def __init__(self, n=16, es=1):
        if n < 3:
            raise ValueError(
                f"n (posit bit-width) must be >= 3 (need at least a sign "
                f"bit and a 2-bit regime), got n={n}."
            )
        if es < 0:
            raise ValueError(f"es (exponent size) must be >= 0, got es={es}.")
        self.n = n
        self.es = es

    @staticmethod
    def _regime_len(regime: int) -> int:
        """Length in bits of the unary-coded regime (including terminator)."""
        return (regime + 2) if regime >= 0 else (-regime + 1)

    @staticmethod
    def _regime_bits(regime: int) -> str:
        return ("1" * (regime + 1) + "0") if regime >= 0 else ("0" * (-regime) + "1")

    def _round_from(self, regime: int, position: float) -> int:
        """
        Iteratively round a (regime, position) pair down to a final
        (n-1)-bit posit magnitude pattern, propagating any rounding carry
        into the regime field. ``position`` is the continuous exponent+
        fraction value in [0, useed_exp) still to be encoded/rounded.
        """
        n, es = self.n, self.es
        useed_exp = 1 << es

        for _ in range(4):
            rlen = self._regime_len(regime)
            avail = (n - 1) - rlen

            if avail < 0:
                # Regime code itself doesn't fit -> clamp to minpos/maxpos.
                # Posits never round a nonzero value down to exact zero.
                return ((1 << (n - 1)) - 1) if regime >= 0 else 1

            if avail == 0:
                # No room for exponent/fraction bits at all: round the
                # position to whichever end of this regime step it's closer
                # to. On an exact tie, break to whichever bit pattern is
                # even (round-half-to-even).
                half = useed_exp / 2
                if position > half:
                    regime += 1
                    position -= useed_exp
                    continue
                if position < half:
                    return int(self._regime_bits(regime)[: n - 1].ljust(n - 1, "0"), 2)
                stay_bits = int(self._regime_bits(regime)[: n - 1].ljust(n - 1, "0"), 2)
                carry_bits = self._round_from(regime + 1, position - useed_exp)
                return stay_bits if stay_bits % 2 == 0 else carry_bits

            # Round the combined exponent+fraction "position" to `avail`
            # bits as one fixed-point value, so any rounding carry rolls
            # naturally from the fraction into the exponent bits.
            scaled = position / useed_exp * (1 << avail)
            raw = int(round(scaled))
            if raw >= (1 << avail):
                # Carried past this regime step entirely -> bump regime and
                # re-round from scratch at the new (possibly wider) field.
                regime += 1
                position -= useed_exp  # exponent is fixed; regime moved up
                continue
            # A tiny negative value can appear right after a carry (position
            # just crossed 0 from the subtraction above); it belongs to this
            # regime step with a zero field, not a further carry.
            raw = max(raw, 0)

            field_bits = format(raw, f"0{avail}b")
            bitstring = (self._regime_bits(regime) + field_bits)[: n - 1].ljust(n - 1, "0")
            return int(bitstring, 2)

        # Should not be reached in practice; fail safe to maxpos.
        return (1 << (n - 1)) - 1

    def _encode_magnitude(self, magnitude: float) -> int:
        """
        Encode a positive float magnitude into an (n-1)-bit posit magnitude
        pattern, with correct round-to-nearest (ties-to-even) carry
        propagation through the exponent and regime fields - the way a
        hardware posit encoder does it, rather than truncating each field
        independently.
        """
        es = self.es
        useed_exp = 1 << es

        # Exact base-2 decomposition (avoids log2 floating-point error near
        # boundaries): magnitude = mantissa * 2**bexp, 0.5 <= mantissa < 1.
        mantissa, bexp = math.frexp(magnitude)
        exponent = bexp - 1                 # magnitude = 2**exponent * (1+frac_full)
        frac_full = mantissa * 2 - 1        # in [0, 1)

        regime = exponent // useed_exp       # floor division (correct for negatives)
        exp_remainder = exponent % useed_exp  # in [0, useed_exp)

        # Continuous "position" within this regime step, in [0, useed_exp).
        # This combines the integer exponent-field value and the fractional
        # part into one quantity we round as a whole, so a rounding carry
        # naturally rolls over into the regime instead of being lost.
        position = exp_remainder + frac_full

        return self._round_from(regime, position)

    def _decode_magnitude(self, bit_int: int) -> float:
        """Decode an (n-1)-bit unsigned posit magnitude pattern back to a float."""
        n, es = self.n, self.es
        bits = format(bit_int, f"0{n - 1}b")

        # Read unary regime run (a run of same bits, terminated by the
        # opposite bit - the terminator itself is also consumed here).
        first = bits[0]
        i = 0
        while i < len(bits) and bits[i] == first:
            i += 1
        if first == "1":
            regime = i - 1
        else:
            regime = -i
        pos = i + 1 if i < len(bits) else i  # skip the terminator bit too

        exp_len = min(es, len(bits) - pos)
        exp_bits = bits[pos:pos + exp_len]
        exp_val = int(exp_bits, 2) if exp_bits else 0
        pos += exp_len

        frac_bits = bits[pos:]
        frac_len = len(frac_bits)
        frac_val = int(frac_bits, 2) / (1 << frac_len) if frac_len > 0 else 0.0

        k = regime * (2 ** es) + exp_val
        return (2.0 ** k) * (1.0 + frac_val)

    def float_to_posit_bits(self, value: float) -> int:
        """Convert a single float to an n-bit posit integer representation."""
        n = self.n

        if value == 0:
            return 0
        if np.isnan(value) or np.isinf(value):
            return 1 << (n - 1)  # NaR (Not a Real)

        sign = 1 if value < 0 else 0
        magnitude = abs(value)

        value_bits = self._encode_magnitude(magnitude)

        if sign:
            posit_int = ((~value_bits) + 1) & ((1 << n) - 1)
        else:
            posit_int = value_bits

        return posit_int & ((1 << n) - 1)

    def convert_array(self, arr: np.ndarray) -> np.ndarray:
        """Vectorized conversion of a numpy array to posit bit patterns."""
        flat = arr.flatten()
        converted = np.array(
            [self.float_to_posit_bits(v) for v in flat], dtype=np.uint32
        )
        return converted.reshape(arr.shape)

    def convert_weights(self, weights_list, biases_list):
        """Convert all layer weights and biases of a trained MLP to posit bits."""
        posit_weights = [self.convert_array(w) for w in weights_list]
        posit_biases = [self.convert_array(b) for b in biases_list]
        return posit_weights, posit_biases


# ---------------------------------------------------------------------------
# 4. Export: LUT (Verilog ROM) or .mem file
# ---------------------------------------------------------------------------
class PositExporter:
    """Exports posit-converted weights/biases as a Verilog LUT module or .mem file."""

    def __init__(self, n=16, output_dir="posit_export"):
        self.n = n
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def to_mem_file(self, posit_array, filename):
        """Write posit values as a plain hex .mem file (one value per line)."""
        path = os.path.join(self.output_dir, filename)
        hex_width = self.n // 4 if self.n % 4 == 0 else (self.n // 4) + 1
        with open(path, "w") as f:
            for v in posit_array.flatten():
                f.write(f"{int(v):0{hex_width}x}\n")
        return path

    def to_lut_verilog(self, posit_array, module_name, filename=None):
        """Write posit values as a synthesizable Verilog ROM/LUT module."""
        filename = filename or f"{module_name}.v"
        path = os.path.join(self.output_dir, filename)
        flat = posit_array.flatten()
        depth = len(flat)
        addr_width = max(1, int(np.ceil(np.log2(depth)))) if depth > 1 else 1

        lines = [
            f"module {module_name} (",
            f"    input  wire [{addr_width - 1}:0] addr,",
            f"    output reg  [{self.n - 1}:0] data",
            ");",
            "",
            "always @(*) begin",
            "    case (addr)",
        ]
        for i, v in enumerate(flat):
            lines.append(f"        {addr_width}'d{i}: data = {self.n}'d{int(v)};")
        lines += [
            f"        default: data = {self.n}'d0;",
            "    endcase",
            "end",
            "",
            "endmodule",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def export(self, posit_array, name, mode="lut"):
        """
        Parameters
        ----------
        mode : str
            'lut' -> synthesizable Verilog LUT/ROM module
            'mem' -> plain .mem hex file (for $readmemh)
        """
        if mode == "lut":
            return self.to_lut_verilog(posit_array, module_name=name)
        elif mode == "mem":
            return self.to_mem_file(posit_array, filename=f"{name}.mem")
        else:
            raise ValueError("mode must be 'lut' or 'mem'")


# ---------------------------------------------------------------------------
# 5. End-to-end Pipeline
# ---------------------------------------------------------------------------
class PositMLPPipeline:
    """
    High-level orchestrator: dataset -> MLP training -> posit conversion
    -> hardware export (LUT or .mem), matching the 4 pipeline stages
    requested for the library's public interface.
    """

    def __init__(self, n=16, es=1, output_dir="posit_export"):
        self.loader = None
        self.builder = None
        self.n = n
        self.es = es
        self.converter = PositConverter(n=n, es=es)
        self.exporter = PositExporter(n=n, output_dir=output_dir)

    # Stage 1: load dataset
    def load_dataset(self, source, target_column=None, normalize=True,
                      test_size=0.2, random_state=42):
        self.loader = DatasetLoader(normalize=normalize, test_size=test_size,
                                     random_state=random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = self.loader.load(
            source, target_column=target_column
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    # Stage 2: choose architecture & train
    def build_and_train(self, hidden_layers=(8, 8), task="classification",
                         activation="relu", max_iter=2000, random_state=42):
        self.builder = MLPBuilder(hidden_layers=hidden_layers, task=task,
                                   activation=activation, max_iter=max_iter,
                                   random_state=random_state)
        self.builder.train(self.X_train, self.y_train)
        score = self.builder.evaluate(self.X_test, self.y_test)
        print(f"[PositMLPPipeline] Trained model accuracy/score: {score:.4f}")
        return self.builder.model

    # Stage 3: convert to posit (n, es given by the user)
    def convert_to_posit(self, n=None, es=None):
        if n is not None or es is not None:
            self.n = n or self.n
            self.es = es or self.es
            self.converter = PositConverter(n=self.n, es=self.es)
            self.exporter = PositExporter(n=self.n, output_dir=self.exporter.output_dir)

        weights, biases = self.builder.get_weights_and_biases()
        self.posit_weights, self.posit_biases = self.converter.convert_weights(weights, biases)
        print(f"[PositMLPPipeline] Converted weights/biases to Posit{self.n}, es={self.es}.")
        return self.posit_weights, self.posit_biases

    # Stage 4a: ask the user LUT vs mem interactively
    def export_interactive(self):
        """Prompt the user: LUT (Verilog) or .mem file?"""
        choice = input(
            "Export format - type 'lut' for a Verilog LUT module, "
            "or 'mem' for a .mem hex file: "
        ).strip().lower()
        return self.export(mode=choice)

    # Stage 4b: export programmatically
    def export(self, mode="lut"):
        paths = []
        for i, w in enumerate(self.posit_weights):
            paths.append(self.exporter.export(w, name=f"layer{i}_weights", mode=mode))
        for i, b in enumerate(self.posit_biases):
            paths.append(self.exporter.export(b, name=f"layer{i}_biases", mode=mode))
        print(f"[PositMLPPipeline] Exported {len(paths)} files to "
              f"'{self.exporter.output_dir}' (mode={mode}).")
        return paths


if __name__ == "__main__":
    # Quick smoke test / usage example
    pipeline = PositMLPPipeline(n=16, es=1)
    pipeline.load_dataset("iris")
    pipeline.build_and_train(hidden_layers=(8, 8), task="classification")
    pipeline.convert_to_posit()
    pipeline.export(mode="lut")
