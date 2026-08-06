"""Minimal reader for legacy compressed SPSS .sav files.

This fallback is intentionally limited to the legacy $FL2 format used by
``merged_all.sav`` in this project. It supports numeric variables, short string
variables (one 8-byte slot), value labels, declared user-missing values, and
SPSS byte-code compression. When pyreadstat is installed, downstream scripts
prefer pyreadstat and use this module only as a portability fallback.
"""
from __future__ import annotations

from pathlib import Path
import math
import struct
from typing import Any

import pandas as pd


def read_legacy_sav(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as handle:
        header = handle.read(176)
        if header[:4] != b"$FL2":
            raise ValueError(f"Unsupported SPSS file signature in {path}")

        layout, case_size, compression, weight_index, n_cases = struct.unpack(
            "<5i", header[64:84]
        )
        bias = struct.unpack("<d", header[84:92])[0]

        variables: list[dict[str, Any]] = []
        value_label_sets: list[tuple[list[tuple[bytes, str]], list[int]]] = []

        def read_i32() -> int:
            raw = handle.read(4)
            if len(raw) != 4:
                raise EOFError("Unexpected end of SPSS dictionary")
            return struct.unpack("<i", raw)[0]

        while True:
            record_type = read_i32()
            if record_type == 2:
                variable_type = read_i32()
                has_label = read_i32()
                n_missing = read_i32()
                print_format = read_i32()
                write_format = read_i32()
                name = handle.read(8).decode("latin1").rstrip(" \x00")

                label = None
                if has_label:
                    label_length = read_i32()
                    label = handle.read(label_length).decode("latin1", "replace")
                    handle.read((-label_length) % 4)

                missing_values: list[float | str] = []
                for _ in range(abs(n_missing)):
                    raw = handle.read(8)
                    if variable_type == 0:
                        missing_values.append(struct.unpack("<d", raw)[0])
                    else:
                        missing_values.append(raw.decode("latin1", "replace").rstrip(" \x00"))

                variables.append(
                    {
                        "name": name,
                        "type": variable_type,
                        "label": label,
                        "n_missing": n_missing,
                        "missing_values": missing_values,
                        "print_format": print_format,
                        "write_format": write_format,
                    }
                )

            elif record_type == 3:
                n_labels = read_i32()
                entries: list[tuple[bytes, str]] = []
                for _ in range(n_labels):
                    raw_value = handle.read(8)
                    label_length = handle.read(1)[0]
                    label = handle.read(label_length).decode("latin1", "replace")
                    handle.read((-(1 + label_length)) % 8)
                    entries.append((raw_value, label))

                if read_i32() != 4:
                    raise ValueError("Malformed SPSS value-label index record")
                n_variables = read_i32()
                indices = [read_i32() for _ in range(n_variables)]
                value_label_sets.append((entries, indices))

            elif record_type == 6:
                handle.read(read_i32() * 80)

            elif record_type == 7:
                _subtype = read_i32()
                size = read_i32()
                count = read_i32()
                handle.read(size * count)

            elif record_type == 999:
                _filler = read_i32()
                break

            else:
                raise ValueError(f"Unsupported SPSS dictionary record type: {record_type}")

        if len(variables) != case_size:
            raise ValueError(
                f"SPSS case size is {case_size}, but {len(variables)} variable slots were found"
            )

        for entries, indices in value_label_sets:
            for one_based_index in indices:
                variable = variables[one_based_index - 1]
                mapping: dict[float | str, str] = {}
                for raw_value, label in entries:
                    if variable["type"] == 0:
                        key: float | str = struct.unpack("<d", raw_value)[0]
                    else:
                        key = raw_value.decode("latin1", "replace").rstrip(" \x00")
                    mapping[key] = label
                variable["value_labels"] = mapping

        rows: list[list[float | str]] = []

        if compression == 0:
            for _ in range(n_cases):
                row: list[float | str] = []
                for variable in variables:
                    raw = handle.read(8)
                    if variable["type"] == 0:
                        row.append(struct.unpack("<d", raw)[0])
                    else:
                        row.append(raw.decode("latin1", "replace").rstrip(" \x00"))
                rows.append(row)

        elif compression == 1:
            current_row: list[float | str] = []
            while len(rows) < n_cases:
                codes = handle.read(8)
                if len(codes) != 8:
                    raise EOFError(
                        f"Unexpected end of compressed SPSS data after {len(rows)} cases"
                    )

                raw_payloads = [handle.read(8) for code in codes if code == 253]
                raw_iter = iter(raw_payloads)
                reached_end = False

                for code in codes:
                    if code == 0:
                        continue
                    if code == 252:
                        reached_end = True
                        break

                    variable = variables[len(current_row)]
                    if code == 253:
                        raw = next(raw_iter)
                        if variable["type"] == 0:
                            value: float | str = struct.unpack("<d", raw)[0]
                        else:
                            value = raw.decode("latin1", "replace").rstrip(" \x00")
                    elif code == 254:
                        value = "" if variable["type"] > 0 else math.nan
                    elif code == 255:
                        value = math.nan if variable["type"] == 0 else ""
                    else:
                        value = float(code - bias) if variable["type"] == 0 else str(code - bias)

                    current_row.append(value)
                    if len(current_row) == case_size:
                        rows.append(current_row)
                        current_row = []
                        if len(rows) == n_cases:
                            break

                if reached_end:
                    break
        else:
            raise NotImplementedError(
                f"Compression mode {compression} is not supported by the fallback reader"
            )

    dataframe = pd.DataFrame(rows, columns=[variable["name"] for variable in variables])
    metadata = {
        "layout": layout,
        "case_size": case_size,
        "compression": compression,
        "weight_index": weight_index,
        "n_cases": n_cases,
        "bias": bias,
        "variables": variables,
    }
    return dataframe, metadata


def read_sav_portable(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an SPSS file with pyreadstat when available, otherwise fallback."""
    try:
        import pyreadstat  # type: ignore
    except ImportError:
        return read_legacy_sav(path)

    dataframe, metadata = pyreadstat.read_sav(str(path), apply_value_formats=False)
    variables = []
    labels = getattr(metadata, "column_names_to_labels", {}) or {}
    missing_ranges = getattr(metadata, "missing_ranges", {}) or {}
    value_labels = getattr(metadata, "variable_value_labels", {}) or {}
    for name in dataframe.columns:
        declared_missing: list[float | str] = []
        for item in missing_ranges.get(name, []):
            if item.get("lo") == item.get("hi"):
                declared_missing.append(item.get("lo"))
        variables.append(
            {
                "name": name,
                "label": labels.get(name),
                "missing_values": declared_missing,
                "value_labels": value_labels.get(name, {}),
            }
        )
    return dataframe, {"variables": variables, "reader": "pyreadstat"}
