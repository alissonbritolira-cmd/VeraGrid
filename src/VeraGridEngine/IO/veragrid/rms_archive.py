# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Read-only compatibility for the first published RMS archive prototype.

New files use the existing developed result_store/result_index format instead.
This reader remains only so files created by the earlier public prototype can
be reopened and exported. It is not an alternative active storage backend.
"""
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
from VeraGridEngine.IO.veragrid.result_store import ChunkedArray, ResultReader


MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CHUNK_ROWS = 131072
FORMAT = "VeraGrid.RmsColumns"


def _identity(path):
    """Detect replacement instead of reading a different archive silently."""
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns




class ZipRmsValues:
    """Read-only array-like RMS tensor with basic NumPy integer/slice indexing.

    Each request opens and closes its own ZIP handle. Full ``np.asarray`` is an
    explicit materialization; normal GUI channel selection reads one column.
    Returned arrays are independent and may safely be edited by a caller.
    """
    def __init__(self, filename, member, archive=None):
        self.filename = str(Path(filename).resolve())
        self.member = member
        self.identity = _identity(self.filename)
        self.members_read = 0
        self.bytes_read = 0
        if archive is None:
            with zipfile.ZipFile(self.filename) as opened:
                self._read_manifest(opened)
        else:
            self._read_manifest(archive)

    def _read_manifest(self, archive):
        info = archive.getinfo(self.member)
        if info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError("RMS index exceeds 16 MiB")
        manifest = json.loads(archive.read(info))
        if (manifest.get("format") != FORMAT or type(manifest.get("version")) is not int
                or manifest.get("version") != 1):
            raise ValueError("Unsupported RMS archive format or version")
        shape = manifest.get("shape")
        rows = manifest.get("chunk_rows")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(type(n) is not int or n < 0 for n in shape)
                or type(rows) is not int or not 1 <= rows <= MAX_CHUNK_ROWS):
            raise ValueError("Invalid RMS shape or chunk row count")
        self.shape = tuple(shape)
        self.ndim = 3
        self.size = math.prod(shape)
        if not isinstance(manifest.get("dtype"), str):
            raise ValueError("Missing RMS numeric dtype")
        self.dtype = np.dtype(manifest["dtype"])
        if self.dtype.kind != "f" or self.dtype.itemsize != 8:
            raise ValueError("Indexed RMS traces require float64 values")
        self.chunk_rows = rows
        expected = math.ceil(shape[0] / rows) * shape[1] * shape[2]
        blocks = manifest.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != expected or expected > 50000:
            raise ValueError("Incomplete or oversized RMS index")
        all_names = archive.namelist()
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate archive members are ambiguous")
        self.blocks = {}
        prefix = None
        for block in blocks:
            name = block.get("member", "")
            match = re.fullmatch(r"rms_results/([0-9a-f]{32})/g(\d+)/v(\d+)/t(\d+)\.npy", name)
            digest = block.get("sha256", "")
            if match is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Invalid RMS chunk member or digest")
            current_prefix, group, variable, start = match.groups()
            group, variable, start = int(group), int(variable), int(start)
            if prefix is None:
                prefix = current_prefix
            if (prefix != current_prefix or group >= shape[2] or variable >= shape[1]
                    or start >= shape[0] or start % rows):
                raise ValueError("RMS chunk is outside its declared tensor")
            key = group, variable, start
            if key in self.blocks:
                raise ValueError("Duplicate RMS chunk coordinates")
            try:
                chunk_info = archive.getinfo(name)
            except KeyError as exc:
                raise ValueError("An indexed RMS chunk is missing") from exc
            expected_bytes = min(rows, shape[0] - start) * self.dtype.itemsize
            if not expected_bytes < chunk_info.file_size <= expected_bytes + 4096:
                raise ValueError("Unexpected RMS chunk byte size")
            self.blocks[key] = name, digest

    def _check_identity(self):
        if _identity(self.filename) != self.identity:
            raise ValueError("The RMS archive changed; reopen its saved results")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if sum(item is Ellipsis for item in key) > 1:
            raise IndexError("Only one ellipsis is allowed")
        if any(item is Ellipsis for item in key):
            position = next(i for i, item in enumerate(key) if item is Ellipsis)
            key = key[:position] + (slice(None),) * (4 - len(key)) + key[position + 1:]
        key += (slice(None),) * (3 - len(key))
        if len(key) != 3:
            raise IndexError("RMS values have three axes")
        indices, scalar_axes = [], []
        for axis, (item, size) in enumerate(zip(key, self.shape)):
            if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
                index = int(item)
                if index < 0:
                    index += size
                if not 0 <= index < size:
                    raise IndexError("RMS index out of range")
                indices.append(range(index, index + 1))
                scalar_axes.append(axis)
            elif isinstance(item, slice):
                indices.append(range(*item.indices(size)))
            else:
                raise IndexError("Indexed RMS values support integers and slices only")
        output = np.empty(tuple(len(index) for index in indices), dtype=self.dtype)
        self._check_identity()
        if output.size:
            selected_rows = np.arange(indices[0].start, indices[0].stop,
                                      indices[0].step, dtype=np.int64)
            ascending = selected_rows if indices[0].step > 0 else selected_rows[::-1]
            first_chunk = int(ascending[0]) // self.chunk_rows * self.chunk_rows
            last_chunk = int(ascending[-1]) // self.chunk_rows * self.chunk_rows
            with zipfile.ZipFile(self.filename) as archive:
                self._check_identity()
                for group_out, group in enumerate(indices[2]):
                    for variable_out, variable in enumerate(indices[1]):
                        for start in range(first_chunk, last_chunk + 1, self.chunk_rows):
                            lower = int(np.searchsorted(ascending, start))
                            upper = int(np.searchsorted(ascending, start + self.chunk_rows))
                            if lower == upper:
                                continue
                            destinations = (np.arange(lower, upper) if indices[0].step > 0
                                            else len(selected_rows) - 1 - np.arange(lower, upper))
                            offsets = ascending[lower:upper] - start
                            name, digest = self.blocks[group, variable, start]
                            payload = archive.read(name)
                            if hashlib.sha256(payload).hexdigest() != digest:
                                raise ValueError("RMS chunk SHA-256 does not match its index")
                            expected = min(self.chunk_rows, self.shape[0] - start)
                            stream = BytesIO(payload)
                            version = np.lib.format.read_magic(stream)
                            if version == (1, 0):
                                shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=2048)
                            elif version == (2, 0):
                                shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=2048)
                            else:
                                raise ValueError("Unsupported RMS NPY chunk version")
                            if (shape != (expected,) or dtype != self.dtype
                                    or len(payload) - stream.tell() != expected * self.dtype.itemsize):
                                raise ValueError("RMS chunk shape or dtype does not match its index")
                            # Validate the header before allocating from an untrusted shape.
                            column = np.frombuffer(payload, dtype=dtype, offset=stream.tell(), count=expected)
                            self.members_read += 1
                            self.bytes_read += len(payload)
                            output[destinations, variable_out, group_out] = column[offsets]
            self._check_identity()
        return np.squeeze(output, axis=tuple(scalar_axes))

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("Indexed RMS values cannot provide a zero-copy dense tensor")
        result = self[:, :, :]
        return result.astype(dtype, copy=False) if dtype is not None else result


def rebind_rms_archives(sessions, filename, folder="sessions"):
    """Rebind lazy sources only after the atomic save has replaced its target."""
    for session in sessions or ():
        results = session.results
        values = getattr(results, "values", None)
        if isinstance(values, (ChunkedArray, ZipRmsValues)):
            prefix = f"{folder}/{session.name}/{session.tpe.value}"
            with zipfile.ZipFile(filename) as archive:
                if prefix + '/result_index.json' in archive.namelist():
                    # Native commit callbacks already retarget ChunkedArray in place.
                    if isinstance(values, ChunkedArray):
                        continue
                    index = json.loads(archive.read(prefix + '/result_index.json'))
                    reader = ResultReader(filename)
                    descriptor = index['arrays']['values']
                    reader.validate_members(descriptor['blocks'])
                    results.values = ChunkedArray(reader, descriptor)
                else:
                    # The user explicitly selected the historical dense format.
                    with archive.open(prefix + '/values.npy') as stream:
                        results.values = np.load(stream, allow_pickle=False)


def would_discard_rms_source(current_sessions, saved_sessions, filename):
    """Tell the GUI whether an overwrite would remove its current lazy curves."""
    target = Path(filename).resolve()
    included_results = {id(session.results) for session in saved_sessions or ()}
    for session in current_sessions:
        results = session.results
        values = getattr(results, "values", None)
        source = (values.reader.filename if isinstance(values, ChunkedArray)
                  else values.filename if isinstance(values, ZipRmsValues) else None)
        if (source is not None and Path(source) == target
                and id(results) not in included_results):
            return True
    return False
