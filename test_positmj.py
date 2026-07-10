"""
Test suite for POSITMJ.

Run with:
    pip install pytest --break-system-packages
    pytest tests/ -v

Notes
-----
- The posit-encoder correctness tests are self-contained (round-trip
  decode/encode checks) and do not require any external reference library.
- If the optional `softposit` package is installed, an extra parametrized
  test cross-checks POSITMJ's encoder bit-for-bit against SoftPosit's
  reference C implementation for the three standard posit configurations
  (posit8/es=0, posit16/es=1, posit32/es=2). This test is skipped
  automatically if softposit isn't installed.
"""
import os
import sys
import math
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from POSITMJ import (
    DatasetLoader,
    MLPBuilder,
    PositConverter,
    PositExporter,
    PositMLPPipeline,
)


# ---------------------------------------------------------------------------
# PositConverter - core correctness
# ---------------------------------------------------------------------------
class TestPositConverter:

    def test_zero_maps_to_zero(self):
        conv = PositConverter(n=16, es=1)
        assert conv.float_to_posit_bits(0.0) == 0

    def test_nan_and_inf_map_to_nar(self):
        conv = PositConverter(n=16, es=1)
        nar = 1 << (conv.n - 1)
        assert conv.float_to_posit_bits(float("nan")) == nar
        assert conv.float_to_posit_bits(float("inf")) == nar
        assert conv.float_to_posit_bits(float("-inf")) == nar

    def test_one_and_minus_one(self):
        conv = PositConverter(n=16, es=1)
        # posit16(es=1) representation of 1.0 is 0x4000, -1.0 is 0xC000
        assert conv.float_to_posit_bits(1.0) == 0x4000
        assert conv.float_to_posit_bits(-1.0) == 0xC000

    def test_never_rounds_nonzero_to_zero(self):
        """Posits never flush a nonzero value to exact zero, no matter how tiny."""
        conv = PositConverter(n=8, es=0)
        for tiny in (1e-30, -1e-30, 1e-300, -1e-300):
            bits = conv.float_to_posit_bits(tiny)
            assert bits != 0

    @pytest.mark.parametrize("n,es", [(8, 0), (16, 1), (16, 0), (32, 2), (10, 1)])
    def test_round_trip_all_bit_patterns(self, n, es):
        """Every representable magnitude, decoded then re-encoded, must
        return to the same bit pattern (exhaustive for small n, sampled
        for larger n)."""
        conv = PositConverter(n=n, es=es)
        max_mag_bits = 1 << (n - 1)

        if max_mag_bits <= 2048:
            magnitude_bits_to_test = range(1, max_mag_bits)
        else:
            random.seed(0)
            magnitude_bits_to_test = random.sample(range(1, max_mag_bits), 500)

        for mag_bits in magnitude_bits_to_test:
            value = conv._decode_magnitude(mag_bits)
            re_encoded = conv.float_to_posit_bits(value)
            assert re_encoded == mag_bits, (
                f"n={n} es={es}: magnitude bits {mag_bits} decoded to "
                f"{value!r} but re-encoded to {re_encoded}"
            )

    def test_negative_values_are_twos_complement(self):
        conv = PositConverter(n=16, es=1)
        pos_bits = conv.float_to_posit_bits(2.6)
        neg_bits = conv.float_to_posit_bits(-2.6)
        assert neg_bits == ((~pos_bits) + 1) & 0xFFFF

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            PositConverter(n=2, es=0)

    def test_invalid_es_raises(self):
        with pytest.raises(ValueError):
            PositConverter(n=16, es=-1)

    def test_convert_array_shape_preserved(self):
        conv = PositConverter(n=16, es=1)
        arr = np.array([[1.0, -2.0], [0.5, 3.25]])
        result = conv.convert_array(arr)
        assert result.shape == arr.shape

    def test_convert_weights_matches_layer_count(self):
        conv = PositConverter(n=16, es=1)
        weights = [np.array([1.0, 2.0]), np.array([3.0])]
        biases = [np.array([0.1]), np.array([0.2])]
        pw, pb = conv.convert_weights(weights, biases)
        assert len(pw) == len(weights)
        assert len(pb) == len(biases)


# ---------------------------------------------------------------------------
# PositConverter - optional cross-check against SoftPosit reference
# ---------------------------------------------------------------------------
def _get_softposit_bits(posit_ctor, value):
    """Capture SoftPosit's C-level printf output of a value's raw bits."""
    r, w = os.pipe()
    stdout_fd = os.dup(1)
    os.dup2(w, 1)
    os.close(w)
    try:
        posit_ctor(value).toBinary()
        sys.stdout.flush()
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)
    data = os.read(r, 4096).decode()
    os.close(r)
    return int(data.strip().replace(" ", ""), 2)


class TestAgainstSoftPositReference:
    """Optional: skipped automatically if `softposit` isn't installed."""

    @pytest.fixture(autouse=True)
    def _softposit(self):
        pytest.importorskip("softposit")
        import softposit as sp
        self.sp = sp

    @pytest.mark.parametrize("ctor_name,n,es", [
        ("posit8", 8, 0),
        ("posit16", 16, 1),
        ("posit32", 32, 2),
    ])
    def test_fuzz_matches_softposit(self, ctor_name, n, es):
        conv = PositConverter(n=n, es=es)
        ctor = getattr(self.sp, ctor_name)
        random.seed(123)
        mismatches = []
        for _ in range(500):
            exp = random.uniform(-30, 30)
            v = random.choice([1, -1]) * (2 ** exp) * random.uniform(0.5, 2)
            ref = _get_softposit_bits(ctor, v)
            ours = conv.float_to_posit_bits(v)
            if ref != ours:
                mismatches.append((v, ref, ours))
        assert not mismatches, f"{ctor_name}: {len(mismatches)} mismatches, e.g. {mismatches[:5]}"


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------
class TestDatasetLoader:

    def test_load_builtin_iris(self):
        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.load("iris")
        assert X_train.shape[1] == 4
        assert len(np.unique(y_train)) <= 3

    def test_invalid_source_raises(self):
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.load("not_a_real_dataset")

    def test_csv_without_target_column_raises(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.load(str(csv_path))

    def test_csv_missing_file_raises(self):
        loader = DatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/to/data.csv", target_column="label")

    def test_csv_with_bad_target_column_raises(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
        loader = DatasetLoader()
        with pytest.raises(ValueError):
            loader.load(str(csv_path), target_column="not_a_column")

    def test_valid_external_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        rows = ["a,b,label"]
        for i in range(20):
            rows.append(f"{i},{i * 2},{i % 3}")
        csv_path.write_text("\n".join(rows))
        loader = DatasetLoader(test_size=0.25)
        X_train, X_test, y_train, y_test = loader.load(str(csv_path), target_column="label")
        assert X_train.shape[1] == 2


# ---------------------------------------------------------------------------
# MLPBuilder
# ---------------------------------------------------------------------------
class TestMLPBuilder:

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            MLPBuilder(task="not_a_task")

    def test_invalid_hidden_layers_raises(self):
        with pytest.raises(ValueError):
            MLPBuilder(hidden_layers=())
        with pytest.raises(ValueError):
            MLPBuilder(hidden_layers=(8, -1))

    def test_train_on_iris(self):
        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.load("iris")
        builder = MLPBuilder(hidden_layers=(8, 8), task="classification", max_iter=500)
        builder.train(X_train, y_train)
        acc = builder.evaluate(X_test, y_test)
        assert 0.0 <= acc <= 1.0

    def test_mismatched_lengths_raise(self):
        builder = MLPBuilder()
        with pytest.raises(ValueError):
            builder.train(np.zeros((5, 3)), np.zeros(4))

    def test_nan_in_data_raises(self):
        builder = MLPBuilder()
        X = np.zeros((5, 3))
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            builder.train(X, np.zeros(5))

    def test_evaluate_before_train_raises(self):
        builder = MLPBuilder()
        with pytest.raises(RuntimeError):
            builder.evaluate(np.zeros((2, 2)), np.zeros(2))

    def test_get_weights_before_train_raises(self):
        builder = MLPBuilder()
        with pytest.raises(RuntimeError):
            builder.get_weights_and_biases()


# ---------------------------------------------------------------------------
# PositExporter
# ---------------------------------------------------------------------------
class TestPositExporter:

    def test_mem_file_written(self, tmp_path):
        exporter = PositExporter(n=16, output_dir=str(tmp_path))
        arr = np.array([1, 2, 3], dtype=np.uint32)
        path = exporter.export(arr, name="test_layer", mode="mem")
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 3

    def test_lut_verilog_written(self, tmp_path):
        exporter = PositExporter(n=16, output_dir=str(tmp_path))
        arr = np.array([1, 2, 3], dtype=np.uint32)
        path = exporter.export(arr, name="test_layer", mode="lut")
        assert os.path.exists(path)
        content = open(path).read()
        assert "module test_layer" in content
        assert "endmodule" in content

    def test_invalid_mode_raises(self, tmp_path):
        exporter = PositExporter(n=16, output_dir=str(tmp_path))
        arr = np.array([1], dtype=np.uint32)
        with pytest.raises(ValueError):
            exporter.export(arr, name="x", mode="not_a_mode")


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
class TestPositMLPPipeline:

    def test_full_pipeline_iris(self, tmp_path):
        pipeline = PositMLPPipeline(n=16, es=1, output_dir=str(tmp_path))
        pipeline.load_dataset("iris")
        pipeline.build_and_train(hidden_layers=(8, 8), task="classification", max_iter=500)
        pipeline.convert_to_posit()
        paths = pipeline.export(mode="mem")
        assert len(paths) > 0
        for p in paths:
            assert os.path.exists(p)

    def test_pipeline_with_external_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        rows = ["f1,f2,f3,label"]
        for i in range(30):
            rows.append(f"{i * 0.1},{i * 0.2},{i * 0.3},{i % 3}")
        csv_path.write_text("\n".join(rows))

        out_dir = tmp_path / "export"
        pipeline = PositMLPPipeline(n=16, es=1, output_dir=str(out_dir))
        pipeline.load_dataset(str(csv_path), target_column="label")
        pipeline.build_and_train(hidden_layers=(16, 8), task="classification", max_iter=500)
        pipeline.convert_to_posit()
        paths = pipeline.export(mode="lut")
        assert len(paths) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
