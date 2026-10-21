# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Lossless, bounded regular RMS trajectories owned by an unfinished study.

These files are primary recovery data, never a disposable cache. A native input
snapshot and explicit coverage accompany the committed Float64 tiles. Opening a
study does not create a store. No solver, event or model equation is defined here.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import shutil
import struct
import tempfile
import threading
import time
import uuid
import zlib

import numpy as np


SCHEMA_VERSION = 1
BUFFER_BYTES = 32 * 1024**2
CACHE_BYTES = 32 * 1024**2
CHANNELS_PER_TILE = 16
MINIMUM_FREE_BYTES = 512 * 1024**2


class TrajectoryStorageError(RuntimeError):
    status = 'trajectory_storage_failure'


def recovery_directory(options):
    selected = getattr(options, 'trajectory_directory', '')
    explicit = selected or os.environ.get('VERAGRID_RMS_RECOVERY_DIR', '')
    if explicit:
        base = Path(explicit)
    else:
        local = (os.environ.get('VERAGRID_HOST_LOCALAPPDATA')
                 or os.environ.get('LOCALAPPDATA') or str(Path.home() / '.local/share'))
        base = Path(local) / 'VeraGrid/RmsRecovery'
    if not base.is_absolute():
        raise TrajectoryStorageError('A pasta de recuperação RMS deve ter caminho absoluto.')
    return base.resolve()


def preflight_storage(options):
    """Check the selected volume before PF/preparation; never switch volumes."""
    if getattr(options, 'trajectory_storage', 'memory') != 'lossless_disk':
        return None
    base = recovery_directory(options)
    probe = base
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    required = MINIMUM_FREE_BYTES + BUFFER_BYTES
    try:
        if base.exists() and not base.is_dir():
            raise OSError('O destino existe e não é uma pasta.')
        free = shutil.disk_usage(probe).free
        if free < required:
            raise OSError(f'Espaço livre {free / 1024**2:.1f} MiB; mínimo de preparação '
                          f'{required / 1024**2:.0f} MiB, além do tamanho das entradas e resultados.')
    except OSError as exc:
        raise TrajectoryStorageError('Recuperação RMS indisponível em ' + str(base) + ': ' + str(exc)) from exc
    return dict(directory=str(base), free_bytes=free, required_before_inputs_bytes=required)


def _tile_key(values):
    header = struct.pack('<QQ', *values.shape)
    return hashlib.sha256(header + values.tobytes(order='C')).hexdigest()


def _encode_tile(values):
    values = np.asarray(values, dtype='<f8', order='C')
    bits = values.view('<u8')
    delta = np.empty_like(bits)
    delta[0] = bits[0]
    delta[1:] = np.bitwise_xor(bits[1:], bits[:-1])
    # Reversible byte planes improve compression without rounding a single bit.
    planes = delta.view(np.uint8).reshape(*values.shape, 8).transpose(2, 1, 0)
    return _tile_key(values), zlib.compress(planes.tobytes(order='C'), level=3)


def _decode_tile(key, payload, rows, columns):
    if not (type(rows) is int and 0 < rows <= 512 and type(columns) is int
            and 0 < columns <= CHANNELS_PER_TILE):
        raise ValueError('Invalid RMS recovery tile dimensions.')
    expected = rows * columns * 8
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected + 1)
    if (len(raw) != expected or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError('Invalid RMS recovery tile compressed size.')
    delta = np.frombuffer(raw, dtype=np.uint8).reshape(8, columns, rows)
    delta = delta.transpose(2, 1, 0).copy().view('<u8').reshape(rows, columns)
    bits = np.bitwise_xor.accumulate(delta, axis=0)
    result = bits.view('<f8')
    if _tile_key(result) != key:
        raise ValueError('RMS recovery tile checksum mismatch.')
    result.flags.writeable = False
    return result


class RmsTrajectoryStore:
    """One study, one bounded writer and immutable committed group views.

    SQLite transactions bind tiles and accepted clocks to the same frontier.
    Recovery can consult that frontier and rerun from the preserved inputs;
    this does not claim a restart of hidden integrator/FMU state mid-trajectory.
    """

    def __init__(self, directory, *, create=False, metadata=None):
        self.directory = Path(directory).resolve()
        self.database = self.directory / 'trajectory.sqlite3'
        self._cache, self._cache_bytes = OrderedDict(), 0
        self._lock = threading.RLock()
        self._writer = None
        self._cancel_reader = None
        self.stats = dict(commits=0, tiles_written=0, unique_tiles=0,
                          decoded_bytes=0, cache_hits=0, commit_s=0.0)
        if create:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._writer = sqlite3.connect(self.database)
            self._writer.execute('PRAGMA journal_mode=WAL')
            self._writer.execute('PRAGMA synchronous=FULL')
            self._writer.executescript('''
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE rms_groups (id TEXT PRIMARY KEY, n_rows INTEGER NOT NULL,
                    n_cols INTEGER NOT NULL, block_rows INTEGER NOT NULL,
                    committed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'not_started');
                CREATE TABLE blobs (hash TEXT PRIMARY KEY, n_rows INTEGER NOT NULL,
                    n_cols INTEGER NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE tiles (group_id TEXT NOT NULL, row_start INTEGER NOT NULL,
                    column_start INTEGER NOT NULL, hash TEXT NOT NULL,
                    PRIMARY KEY(group_id,row_start,column_start));
                CREATE TABLE clocks (group_id TEXT NOT NULL, row_start INTEGER NOT NULL,
                    n_rows INTEGER NOT NULL, hash TEXT NOT NULL, payload BLOB NOT NULL,
                    PRIMARY KEY(group_id,row_start));
                CREATE TABLE large_metadata (key TEXT UNIQUE NOT NULL, payload BLOB NOT NULL);
            ''')
            self.put_metadata('schema_version', SCHEMA_VERSION)
            self.put_metadata('study', dict(metadata or {},
                storage_contract='lossless_float64_tiles_v1', experimental=True,
                recovery_scope='committed_regular_prefix_and_original_inputs; not complete security qualification',
                buffer_max_bytes=BUFFER_BYTES, cache_max_bytes=CACHE_BYTES,
                uncommitted_irregular_records='remain on the running result; recovery does not claim their completeness'))
        else:
            with self.reader() as connection:
                row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                version = None if row is None else json.loads(row[0])
                if type(version) is not int or version != SCHEMA_VERSION:
                    raise ValueError('Unsupported RMS recovery-store schema.')

    @classmethod
    def create(cls, source_grid, options, variables, groups, *, input_identity=None):
        from VeraGridEngine.IO.file_save import FileSave, FileSavingOptions
        from VeraGridEngine.Simulations.Rms.storage_options import options_snapshot
        preflight_storage(options)
        base = recovery_directory(options)
        base.mkdir(parents=True, exist_ok=True)
        store = cls(base / uuid.uuid4().hex, create=True, metadata=dict(
            created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            input_identity=input_identity, options=options_snapshot(options),
            case_idtag=str(source_grid.idtag), case_name=str(source_grid.name),
            variables=[dict(uid=str(v.uid), name=str(v.name)) for v in variables],
            groups=[dict(idtag=str(g.idtag), name=str(g.name)) for g in groups]))
        snapshot = store.directory / 'inputs.veragrid'
        snapshot_started = time.perf_counter()
        try:
            logger = FileSave(source_grid, str(snapshot), options=FileSavingOptions()).save()
            if logger.has_errors() or not snapshot.is_file():
                raise TrajectoryStorageError('Não foi possível preservar as entradas RMS antes da integração.')
            digest = hashlib.sha256()
            with snapshot.open('rb') as stream:
                while chunk := stream.read(1024**2):
                    digest.update(chunk)
            store.put_metadata('input_snapshot', dict(member='inputs.veragrid',
                sha256=digest.hexdigest(), bytes=snapshot.stat().st_size,
                options_source='study.options; native grid snapshot preserves the case and embedded library'))
            preflight_storage(options)
            store.put_metadata('phase', 'ready')
            store.stats['input_snapshot_s'] = time.perf_counter() - snapshot_started
            return store
        except Exception as exc:
            store.close()
            if isinstance(exc, TrajectoryStorageError):
                raise
            raise TrajectoryStorageError('Falha ao preparar a recuperação RMS: ' + str(exc)) from exc

    @contextmanager
    def reader(self):
        if not self.database.is_file():
            raise ValueError('O armazenamento RMS não está disponível: ' + str(self.database))
        connection = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True)
        try:
            yield connection
        finally:
            connection.close()

    def put_metadata(self, key, value):
        if self._writer is None:
            raise TrajectoryStorageError('RMS recovery writer is closed.')
        with self._writer:
            self._writer.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                                 (str(key), json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def create_rows(self, group_id, shape):
        if self._writer is None:
            raise TrajectoryStorageError('RMS recovery writer is closed.')
        values = RmsStoredRows(self, str(group_id), shape)
        with self._writer:
            self._writer.execute('INSERT INTO rms_groups(id,n_rows,n_cols,block_rows) VALUES (?,?,?,?)',
                                 (str(group_id), int(shape[0]), int(shape[1]), values.rows_per_block))
        return values

    def read_json_stream(self, key, max_decoded_bytes=128 * 1024**2):
        """Bound intermediate JSON bytes; refuse oversized recovery explicitly."""
        import io
        with self.reader() as connection:
            row = connection.execute('SELECT rowid FROM large_metadata WHERE key=?', (str(key),)).fetchone()
            if row is None:
                return None
            decoder = zlib.decompressobj()
            total = 0
            with tempfile.TemporaryFile() as temporary:
                with connection.blobopen('large_metadata', 'payload', row[0], readonly=True) as blob:
                    while block := blob.read(65536):
                        while block:
                            decoded = decoder.decompress(block, 1024**2)
                            block = decoder.unconsumed_tail
                            total += len(decoded)
                            if total > max_decoded_bytes:
                                raise ValueError('Metadados de recuperação excedem o leitor limitado a128MiB; os originais foram preservados. É necessário ampliar a leitura incremental, sem descartar estados.')
                            temporary.write(decoded)
                if not decoder.eof or decoder.unused_data:
                    raise ValueError('Metadados de recuperação truncados ou com dados extras.')
                temporary.seek(0)
                with io.TextIOWrapper(temporary, encoding='utf-8') as text:
                    return json.load(text)

    def put_json_stream(self, key, value):
        """Reuse the native JSON codec, compressing tokens without a full string."""
        from VeraGridEngine.IO.veragrid.generic_io_functions import CustomJSONizer
        encoder = CustomJSONizer(ensure_ascii=False, separators=(',', ':'))
        compressor = zlib.compressobj(level=3)
        with tempfile.TemporaryFile(dir=self.directory) as temporary:
            for token in encoder.iterencode(value):
                encoded = compressor.compress(token.encode('utf-8'))
                if encoded:
                    temporary.write(encoded)
            temporary.write(compressor.flush())
            size = temporary.tell()
            temporary.seek(0)
            with self._writer:
                self._writer.execute('INSERT OR REPLACE INTO large_metadata VALUES (?,zeroblob(?))',
                                     (str(key), size))
                row_id = self._writer.execute('SELECT rowid FROM large_metadata WHERE key=?', (str(key),)).fetchone()[0]
                with self._writer.blobopen('large_metadata', 'payload', row_id) as blob:
                    while chunk := temporary.read(1024**2):
                        blob.write(chunk)

    def capture_results(self, results):
        if results is None:
            return
        # Regular matrices stay in tiles. Irregular vectors use the existing
        # encoder one vector at a time; do not build another full list of states.
        payload = dict(metadata=results.archive_dynamic_metadata(),
            irregular={name: getattr(results, name) for name in
                       ('accepted_substeps', 'boundary_left_limits', 'numerical_state_projections')},
            well_initialized=results.well_initialized, converged=results.converged,
            cancelled=getattr(results, 'cancelled', np.zeros(len(results.rms_events_group_idtags), bool)),
            group_names=results.rms_events_group_names,
            group_idtags=results.rms_events_group_idtags)
        self.put_json_stream('result_snapshot', payload)
        with self._writer:
            for i, identity in enumerate(results.rms_events_group_idtags):
                status = results.execution_diagnostics[i].get('status', 'unknown')
                self._writer.execute('UPDATE rms_groups SET status=? WHERE id=?', (status, str(identity)))
        committed = dict(self._writer.execute('SELECT id,committed FROM rms_groups'))
        coverage = [dict(idtag=str(identity), accepted=int(results.valid_samples[i]),
                        committed=int(committed.get(str(identity), 0)))
                    for i, identity in enumerate(results.rms_events_group_idtags)]
        self.put_metadata('recovery_regular_coverage', coverage)
        has_unpersisted_tail = any(row['accepted'] > row['committed'] for row in coverage)
        self.put_metadata('phase', ('metadata_committed_with_unpersisted_regular_tail'
                                   if has_unpersisted_tail else 'result_snapshot_committed'))

    def diagnostics(self, *, include_path=False):
        info = dict(schema_version=SCHEMA_VERSION, experimental=True,
            contract='lossless_float64_tiles_v1', regular_buffer_max_bytes=BUFFER_BYTES,
            cache_max_bytes=CACHE_BYTES, cache_bytes=self._cache_bytes,
            minimum_free_disk_bytes=MINIMUM_FREE_BYTES,
            stats=dict(self.stats), regular_buffers_only=True,
            interpretation='Regular Float64 states are bounded; irregular states and metadata retain their own memory requirements. Recovery files are primary data, not a cache.')
        if include_path:
            info['recovery_directory'] = str(self.directory)
        return info

    def tile(self, connection, key):
        with self._lock:
            if key in self._cache:
                self.stats['cache_hits'] += 1
                self._cache.move_to_end(key)
                return self._cache[key]
        row = connection.execute('SELECT n_rows,n_cols,payload FROM blobs WHERE hash=?', (key,)).fetchone()
        if row is None:
            raise ValueError('Missing committed RMS recovery tile.')
        values = _decode_tile(key, row[2], row[0], row[1])
        with self._lock:
            self.stats['decoded_bytes'] += values.nbytes
            if key not in self._cache:
                while self._cache and self._cache_bytes + values.nbytes > CACHE_BYTES:
                    _, old = self._cache.popitem(last=False)
                    self._cache_bytes -= old.nbytes
                if values.nbytes <= CACHE_BYTES:
                    self._cache[key] = values
                    self._cache_bytes += values.nbytes
        return values

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0


class RmsStoredRows:
    """Solver-owned staging buffer; only explicit accepted clocks are committed."""

    def __init__(self, store, group_id, shape):
        self.store, self.group_id = store, str(group_id)
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 2 or self.shape[0] < 0 or self.shape[1] <= 0:
            raise ValueError('Invalid RMS recovery row shape.')
        self.dtype = np.dtype('<f8')
        self.ndim, self.size = 2, math.prod(self.shape)
        self.nbytes = self.size * 8
        self.rows_per_block = max(1, min(512, BUFFER_BYTES // (8 * self.shape[1])))
        self._buffer = np.empty((self.rows_per_block, self.shape[1]), dtype=self.dtype)
        self._times = np.empty(self.rows_per_block, dtype=self.dtype)
        self.accepted_count = self.committed_count = self._staged_count = 0
        self._first = 0
        self._last_row, self._last_index = None, -1
        self._sealed = False
        self.storage_error = None

    @classmethod
    def from_committed(cls, store, group_id, shape, block_rows, count):
        """Read a frozen committed frontier, allocating no dense/staging matrix."""
        value = cls.__new__(cls)
        value.store, value.group_id = store, str(group_id)
        value.shape, value.dtype, value.ndim = tuple(shape), np.dtype('<f8'), 2
        value.size = math.prod(value.shape); value.nbytes = value.size * 8
        value.rows_per_block = int(block_rows)
        value._buffer = None; value._times = np.empty(0, dtype='<f8')
        value._last_row, value._last_index = None, -1
        value.accepted_count = value.committed_count = value._staged_count = int(count)
        value._first, value._sealed, value.storage_error = int(count), True, None
        return RmsStoredRowsView(value, count)

    def __setitem__(self, selectors, values):
        if self._sealed or not isinstance(selectors, tuple) or len(selectors) != 2:
            raise ValueError('RMS trajectory write outside its active staging phase.')
        row, columns = selectors
        if (not isinstance(row, (int, np.integer)) or columns != slice(None)
                or not self._first <= row < self._first + self.rows_per_block
                or not 0 <= row < self.shape[0] or row > self._staged_count):
            raise IndexError('RMS trajectory accepts sequential whole-row writes only.')
        value = np.asarray(values)
        if value.shape != (self.shape[1],) or value.dtype != self.dtype:
            raise ValueError('RMS staged row must preserve its full Float64 state.')
        self._buffer[row - self._first] = value
        self._staged_count = max(self._staged_count, int(row) + 1)
        self._last_row, self._last_index = value.copy(), int(row)

    def accept_until(self, count, times):
        """Called only after the existing solver accepts a macro sample."""
        count = int(count)
        if self._sealed or not self.accepted_count <= count <= self._staged_count:
            raise ValueError('RMS accepted frontier contradicts staged solver rows.')
        if count > self._first:
            self._times[:count - self._first] = times[self._first:count]
        self.accepted_count = count
        if count - self._first == self.rows_per_block:
            self._flush()

    def _flush(self):
        count = self.accepted_count - self._first
        if not count:
            return
        connection = self.store._writer
        if connection is None:
            raise TrajectoryStorageError('RMS recovery writer is unavailable.')
        started = time.perf_counter()
        written, unique = 0, 0
        try:
            if shutil.disk_usage(self.store.directory).free < MINIMUM_FREE_BYTES:
                raise OSError('Espaço livre abaixo da reserva de512MiB; integração interrompida sem descartar o prefixo aceito.')
            with connection:
                for col in range(0, self.shape[1], CHANNELS_PER_TILE):
                    values = self._buffer[:count, col:col + CHANNELS_PER_TILE]
                    key, payload = _encode_tile(values)
                    cursor = connection.execute('INSERT OR IGNORE INTO blobs VALUES (?,?,?,?)',
                        (key, count, values.shape[1], sqlite3.Binary(payload)))
                    unique += max(0, cursor.rowcount)
                    connection.execute('INSERT INTO tiles VALUES (?,?,?,?)',
                                       (self.group_id, self._first, col, key))
                    written += 1
                clocks = self._times[:count].tobytes()
                connection.execute('INSERT INTO clocks VALUES (?,?,?,?,?)',
                    (self.group_id, self._first, count, hashlib.sha256(clocks).hexdigest(),
                     sqlite3.Binary(zlib.compress(clocks, level=3))))
                connection.execute('UPDATE rms_groups SET committed=?,status=? WHERE id=?',
                                   (self.accepted_count, 'partial', self.group_id))
            self.committed_count = self.accepted_count
            self._first = self.accepted_count
            self.store.stats['commits'] += 1
            self.store.stats['tiles_written'] += written
            self.store.stats['unique_tiles'] += unique
        except (sqlite3.Error, OSError) as exc:
            # The previous transaction is intact. Keep this bounded RAM tail so
            # ordinary native Save can rescue it to another location.
            self.storage_error = str(exc)
            raise TrajectoryStorageError('Falha na gravação da trajetória RMS; último prefixo confirmado: '
                                         + str(self.committed_count) + ' amostras. ' + str(exc)) from exc
        finally:
            self.store.stats['commit_s'] += time.perf_counter() - started

    def accepted_prefix(self, count):
        count = min(int(count), self.accepted_count)
        if not self._sealed:
            if not self.storage_error:
                self._flush()
            self._sealed = True
            if self._first == self.accepted_count:
                self._buffer = None
        return RmsStoredRowsView(self, count)

    @property
    def resident_bytes(self):
        return ((0 if self._buffer is None else self._buffer.nbytes)
                + (0 if self._last_row is None else self._last_row.nbytes)
                + self._times.nbytes)

    def __getitem__(self, item):
        return self._select(item, self.accepted_count)

    def _select(self, item, count):
        if self.store._cancel_reader is not None:
            self.store._cancel_reader()
        selectors = item if isinstance(item, tuple) else (item,)
        selectors += (slice(None),) * (2 - len(selectors))
        if len(selectors) != 2:
            raise IndexError('RMS recovery row selection must have two axes.')
        indices, scalar_axes = [], []
        for axis, (selector, length) in enumerate(zip(selectors, (count, self.shape[1]))):
            if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
                selected = int(selector)
                if selected < 0:
                    selected += length
                if not 0 <= selected < length:
                    raise IndexError('RMS recovery row/channel out of range.')
                indices.append(np.array([selected], dtype=int))
                scalar_axes.append(axis)
            elif isinstance(selector, slice):
                indices.append(np.arange(*selector.indices(length), dtype=int))
            else:
                selected = np.asarray(selector)
                # Native result projections use Cartesian np.ix_ indices.
                if selected.dtype.kind not in 'iu' or selected.ndim > 2:
                    raise IndexError('Unsupported RMS recovery advanced selection.')
                selected = selected.reshape(-1).astype(int, copy=False)
                if np.any(selected < -length) or np.any(selected >= length):
                    raise IndexError('RMS recovery index out of range.')
                indices.append(np.where(selected < 0, selected + length, selected))
        ri, ci = indices
        result = np.empty((len(ri), len(ci)), dtype=self.dtype)
        if result.size:
            remaining = np.ones(len(ri), dtype=bool)
            if self._last_row is not None:
                positions = np.flatnonzero(ri == self._last_index)
                if len(positions):
                    result[positions] = self._last_row[ci]
                    remaining[positions] = False
            if self._buffer is not None:
                positions = np.flatnonzero(remaining & (ri >= self._first))
                if len(positions):
                    result[positions] = self._buffer[np.ix_(ri[positions] - self._first, ci)]
                    remaining[positions] = False
            wanted = np.flatnonzero(remaining)
            if len(wanted):
                with self.store.reader() as connection:
                    for row_start in sorted(set((ri[wanted] // self.rows_per_block) * self.rows_per_block)):
                        positions = wanted[(ri[wanted] >= row_start)
                                           & (ri[wanted] < row_start + self.rows_per_block)]
                        for col_start in sorted(set((ci // CHANNELS_PER_TILE) * CHANNELS_PER_TILE)):
                            columns = np.flatnonzero((ci >= col_start) & (ci < col_start + CHANNELS_PER_TILE))
                            row = connection.execute('SELECT hash FROM tiles WHERE group_id=? AND row_start=? AND column_start=?',
                                (self.group_id, int(row_start), int(col_start))).fetchone()
                            if row is None:
                                raise ValueError('Requested RMS state is not in the committed prefix.')
                            values = self.store.tile(connection, row[0])
                            result[np.ix_(positions, columns)] = values[np.ix_(ri[positions] - row_start, ci[columns] - col_start)]
        return np.squeeze(result, axis=tuple(scalar_axes)) if scalar_axes else result


class RmsStoredRowsView:
    """Independent read-only coverage over a sealed solver group."""
    def __init__(self, rows, count):
        self.rows, self.shape = rows, (int(count), rows.shape[1])
        self.dtype, self.ndim = rows.dtype, 2
        self.size, self.nbytes = math.prod(self.shape), math.prod(self.shape) * 8

    def __len__(self):
        return self.shape[0]

    @property
    def resident_bytes(self):
        return self.rows.resident_bytes

    def accepted_prefix(self, count):
        if not 0 <= count <= self.shape[0]:
            raise ValueError('RMS stored prefix out of accepted bounds.')
        return RmsStoredRowsView(self.rows, count)

    def __getitem__(self, item):
        return self.rows._select(item, self.shape[0])

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError('A complete array requires explicit materialization of RMS recovery tiles.')
        values = self[:, :]
        return values if dtype is None else values.astype(dtype, copy=False)


def accept_trajectory_rows(values, count, times):
    """No numerical work; ndarray solvers keep their existing semantics."""
    if isinstance(values, RmsStoredRows):
        values.accept_until(count, times)


def accepted_trajectory_values(values, count, *, copy=False):
    if isinstance(values, (RmsStoredRows, RmsStoredRowsView)):
        return values.accepted_prefix(count)
    selected = values[:count, :]
    return selected.copy() if copy else selected
