# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""RMS subset extracted from the developed/tested native result store.

The byte cache, array descriptors, 16-channel tiles and verified immutable NPY
copy use the existing implementation. Region records and development-specific
GUI diagnostics are outside this upstream contribution.
"""
from collections import OrderedDict
import copy
import io
import math
from pathlib import Path
import threading
import zipfile
import zlib
import numpy as np

INDEX_VERSION = 1
CACHE_BYTES = 16 * 1024 * 1024
ARRAY_BLOCK_BYTES = 1024 * 1024


def _identity(path):
    st = Path(path).stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


class ResultReader:
    """LRU de bytes descomprimidos, até 16 MiB; sem handles persistentes."""
    def __init__(self, filename, cache_bytes=CACHE_BYTES):
        self.filename = str(Path(filename).resolve())
        try:self.identity = _identity(self.filename)
        except OSError as exc:raise ValueError('Arquivo de resultados indisponível: '+self.filename) from exc
        self.limit = int(cache_bytes)
        self.cache = OrderedDict()
        self.cache_metadata = {}
        self.cache_size = 0
        self.closed = False
        self.lock = threading.RLock()
        self.stats = dict(members=0, compressed_bytes=0, decoded_bytes=0,
                          cache_hits=0, evictions=0, cache_bytes=0, metadata_checks=0)

    def read(self, member, *, expected_size=None, expected_crc32=None):
        try:
            with self.lock:
                if self.closed:
                    raise ValueError('A consulta dos resultados foi encerrada.')
                if _identity(self.filename) != self.identity:
                    self.cache.clear(); self.cache_metadata.clear(); self.cache_size = 0
                    self.stats['cache_bytes'] = 0
                    raise ValueError('O arquivo dos resultados mudou. Reabra o estudo antes de consultá-lo.')
                if member in self.cache:
                    size, crc = self.cache_metadata[member]
                    if (expected_size is not None and size != expected_size) or (expected_crc32 is not None and crc != expected_crc32):
                        raise ValueError('Integridade do bloco diferente do índice.')
                    self.stats['cache_hits'] += 1
                    self.cache.move_to_end(member)
                    return self.cache[member]
                with zipfile.ZipFile(self.filename) as archive:
                    info = archive.getinfo(member)
                    if (expected_size is not None and info.file_size != expected_size) or (expected_crc32 is not None and info.CRC != expected_crc32):
                        raise ValueError('Integridade do bloco diferente do índice.')
                    value = archive.read(info)  # Leitura completa deste bloco, incluindo seu CRC.
                self.stats['members'] += 1
                self.stats['compressed_bytes'] += info.compress_size
                self.stats['decoded_bytes'] += len(value)
                if len(value) <= self.limit:
                    while self.cache and self.cache_size + len(value) > self.limit:
                        old_member, old = self.cache.popitem(last=False)
                        self.cache_metadata.pop(old_member, None)
                        self.cache_size -= len(old); self.stats['evictions'] += 1
                    self.cache[member] = value; self.cache_size += len(value)
                    self.cache_metadata[member] = (info.file_size, info.CRC)
                self.stats['cache_bytes'] = self.cache_size
                return value
        except (OSError, KeyError, zipfile.BadZipFile, EOFError, zlib.error, ValueError) as exc:
            raise ValueError(f"Falha ao ler bloco {member}: {exc}") from exc

    def copy_member(self, archive, member, destination, *, expected_size, expected_crc32):
        """Copia JSON já persistido por blocos, conferindo CRC sem criar sua árvore Python."""
        try:
            with self.lock:
                if self.closed:raise ValueError('A consulta dos resultados foi encerrada.')
                if _identity(self.filename) != self.identity:
                    raise ValueError('O arquivo dos resultados mudou. Reabra o estudo.')
                with zipfile.ZipFile(self.filename) as source:
                    info = source.getinfo(member)
                    if info.file_size != expected_size or info.CRC != expected_crc32:
                        raise ValueError('Integridade do bloco diferente do índice.')
                    with source.open(info) as src, archive.open(destination, 'w', force_zip64=True) as dst:
                        while block := src.read(1024 * 1024):dst.write(block)
                if _identity(self.filename) != self.identity:
                    raise ValueError('O arquivo dos resultados mudou durante a cópia.')
                self.stats['copied_members'] = self.stats.get('copied_members', 0)+1
                self.stats['copied_bytes'] = self.stats.get('copied_bytes', 0)+info.file_size
                return archive.getinfo(destination)
        except (OSError, KeyError, zipfile.BadZipFile, EOFError, zlib.error, ValueError) as exc:
            raise ValueError(f'Falha ao copiar bloco {member}: {exc}') from exc

    def validate_members(self, entries):
        """Presença/tamanho/CRC do diretório central; não lê/descomprime blocos não consultados."""
        with self.lock:
            if self.closed:raise ValueError('A consulta dos resultados foi encerrada.')
            try:identity = _identity(self.filename)
            except OSError as exc:raise ValueError('Arquivo de resultados indisponível: '+self.filename) from exc
            if identity != self.identity:
                raise ValueError('O arquivo dos resultados mudou. Reabra o estudo.')
            if isinstance(entries,(list,tuple)) and not entries:
                # Empty irregular streams have no members to verify. In a
                # large multi-scenario study, opening its central directory
                # again for every empty stream dominated channel restoration.
                # Closed-reader/file identity checks above remain mandatory.
                self.stats['empty_metadata_queries_skipped']=self.stats.get('empty_metadata_queries_skipped',0)+1
                return
            with zipfile.ZipFile(self.filename) as archive:
                for entry in entries:
                    if not isinstance(entry, dict) or not _safe_member(entry.get('member')):
                        raise ValueError('Referência de membro inválida.')
                    try:info = archive.getinfo(entry['member'])
                    except KeyError as exc:raise ValueError('Bloco de resultado ausente: '+entry['member']) from exc
                    if ('nbytes' in entry and info.file_size != entry['nbytes']) or ('crc32' in entry and info.CRC != entry['crc32']):
                        raise ValueError('Integridade do bloco diferente do índice.')
                    self.stats['metadata_checks'] += 1

    def release_cache(self):
        """Libera bytes derivados sem invalidar outra consulta histórica do arquivo."""
        with self.lock:
            self.cache.clear(); self.cache_metadata.clear(); self.cache_size = 0
            self.stats["cache_bytes"] = 0

    def close(self):
        with self.lock:
            self.closed = True; self.cache.clear(); self.cache_metadata.clear(); self.cache_size = 0
            self.stats['cache_bytes'] = 0


    def with_prefix_alias(self, source, destination):
        return _ResultReaderAlias(self, source, destination)


class _ResultReaderAlias:
    """Contexto de um índice remapeado; compartilha somente bytes/lock/stat do mesmo arquivo."""
    def __init__(self, reader, source, destination):
        if not _safe_member(source) or not _safe_member(destination):
            raise ValueError('Prefixo de resultados inválido.')
        self.reader = reader; self.source = source; self.destination = destination

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def read(self, member, **kwargs):
        if member == self.source:
            actual = self.destination
        elif member.startswith(self.source + '/'):
            actual = self.destination + member[len(self.source):]
        else:
            raise ValueError('Resultado fora do prefixo declarado do mesmo estudo.')
        return self.reader.read(actual, **kwargs)

    def copy_member(self, archive, member, destination, **kwargs):
        if not _safe_member(member) or not member.startswith(self.source + '/'):
            raise ValueError('Bloco fora do prefixo do mesmo estudo.')
        actual = self.destination + member[len(self.source):]
        return self.reader.copy_member(archive, actual, destination, **kwargs)

    def validate_members(self, entries):
        mapped = []
        for entry in entries:
            if not isinstance(entry, dict):raise ValueError('Referência de membro inválida.')
            selected = dict(entry);member = selected.get('member')
            if not _safe_member(member) or not member.startswith(self.source + '/'):
                raise ValueError('Bloco fora do prefixo do mesmo estudo.')
            selected['member'] = self.destination + member[len(self.source):];mapped.append(selected)
        return self.reader.validate_members(mapped)

    def with_prefix_alias(self, source, destination):
        return _ResultReaderAlias(self, source, destination)


def _safe_member(member):
    return (isinstance(member, str) and bool(member) and not member.startswith('/')
            and '\\' not in member
            and all(part not in ('', '.', '..') for part in member.split('/')))


class ChunkedArray:
    """Seleção de tempo/canal/cenário lê somente os blocos que se interceptam."""
    def __init__(self, reader, descriptor):
        self.reader = reader; self.descriptor = descriptor
        self.shape = tuple(descriptor['shape']); self.dtype = np.dtype(descriptor['dtype'])
        self.ndim = len(self.shape); self.size = math.prod(self.shape)
        if self.dtype.hasobject or self.ndim != 3 or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError('Forma ou dtype do resultado RMS inválido.')
        blocks = descriptor.get('blocks')
        if not isinstance(blocks, list):raise ValueError('Índice RMS inválido.')
        expected = 0; partitions = {}; members = set()
        for block in blocks:
            if not isinstance(block, dict) or not _safe_member(block.get('member')):
                raise ValueError('Membro de bloco RMS inválido.')
            for key in ('nbytes', 'crc32'):
                if key in block and (type(block[key]) is not int or block[key] < 0):
                    raise ValueError('Metadado de integridade RMS inválido.')
            starts, stops = block['starts'], block['stops']
            if len(starts) != self.ndim or len(stops) != self.ndim or any(
                    type(a) is not int or type(b) is not int or not 0 <= a < b <= n
                    for a,b,n in zip(starts, stops, self.shape)):
                raise ValueError('Intervalo de bloco RMS inválido.')
            if stops[2] != starts[2]+1 or block['member'] in members:
                raise ValueError('Bloco RMS duplicado ou grupo inválido.')
            members.add(block['member'])
            partitions.setdefault((starts[2],starts[1],stops[1]), []).append((starts[0],stops[0]))
            expected += math.prod(b-a for a,b in zip(starts,stops))
        if expected != self.size:raise ValueError('Cobertura de blocos RMS incompleta.')
        if self.size:
            for group in range(self.shape[2]):
                cols = sorted((a,b) for g,a,b in partitions if g == group)
                cursor = 0
                for a,b in cols:
                    if a != cursor:raise ValueError('Sobreposição ou lacuna de canais RMS.')
                    cursor = b; row_cursor = 0
                    for start,stop in sorted(partitions[(group,a,b)]):
                        if start != row_cursor:raise ValueError('Sobreposição ou lacuna temporal RMS.')
                        row_cursor = stop
                    if row_cursor != self.shape[0]:raise ValueError('Bloco RMS com tempo incompleto.')
                if cursor != self.shape[1]:raise ValueError('Grupo RMS com canais incompletos.')

    def __len__(self):return self.shape[0]

    def __getitem__(self, item):
        selectors = item if isinstance(item, tuple) else (item,)
        if any(v is Ellipsis for v in selectors):
            pos = next(i for i,v in enumerate(selectors) if v is Ellipsis)
            selectors = selectors[:pos] + (slice(None),)*(self.ndim-len(selectors)+1) + selectors[pos+1:]
        selectors = selectors + (slice(None),)*(self.ndim-len(selectors))
        if len(selectors) != self.ndim:raise IndexError('Número de dimensões inválido.')
        indices = []; scalar_axes = []
        for axis, (selector, size) in enumerate(zip(selectors, self.shape)):
            if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
                selected = int(selector); selected = selected + size if selected < 0 else selected
                if not 0 <= selected < size:raise IndexError('Índice fora do resultado.')
                indices.append(np.array([selected], dtype=int)); scalar_axes.append(axis)
            elif isinstance(selector, slice):
                indices.append(np.arange(*selector.indices(size), dtype=int))
            elif (axis == 1 and isinstance(selectors[0], slice)
                  and isinstance(selectors[2], (int, np.integer))
                  and not isinstance(selectors[2], (bool, np.bool_))
                  and isinstance(selector, (list, tuple, np.ndarray))):
                # Native channel projection uses values[time_slice, columns,
                # group]. One integer column vector has the same axis layout
                # as this bounded gather; it must not materialize every group.
                # Other advanced NumPy combinations retain the legacy fallback
                # below, including its pairwise indexing/axis-order semantics.
                selected=np.asarray(selector)
                if selected.ndim != 1 or (selected.size and selected.dtype.kind not in 'iu'):
                    return np.asarray(self)[item]
                if np.any(selected < -size) or np.any(selected >= size):
                    raise IndexError('Canal fora do resultado RMS.')
                selected=selected.astype(int,copy=False)
                indices.append(np.where(selected < 0, selected + size, selected))
            else:
                # Preserve NumPy's complete advanced-index semantics as an explicit full read.
                return np.asarray(self)[item]
        output = np.empty(tuple(len(v) for v in indices), dtype=self.dtype)
        if any(len(values) == 0 for values in indices):
            # Metadata validation deliberately requests zero columns. There
            # are no numeric bytes to read and no archive partitions to scan.
            return np.squeeze(output, axis=tuple(scalar_axes)) if scalar_axes else output
        for block in self.descriptor['blocks']:
            positions = [np.flatnonzero((values >= a) & (values < b))
                         for values,a,b in zip(indices,block['starts'],block['stops'])]
            if not all(len(v) for v in positions):continue
            try:
                raw = self.reader.read(block['member'], expected_size=block.get('nbytes'),
                                       expected_crc32=block.get('crc32'))
                expected = tuple(b-a for a,b in zip(block['starts'],block['stops']))
                # Reuse the developed immutable-copy header checks before a
                # selected read can allocate a shape supplied by the file.
                header = io.BytesIO(raw)
                version = np.lib.format.read_magic(header)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(header)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(header)
                else:
                    parser = getattr(np.lib.format, '_read_array_header', None)
                    if parser is None:
                        from numpy.lib._format_impl import _read_array_header as parser
                    shape, _, dtype = parser(header, version)
                if (shape != expected or dtype != self.dtype or dtype.hasobject
                        or len(raw) != header.tell() + math.prod(expected) * self.dtype.itemsize):
                    raise ValueError('Cabeçalho de bloco RMS diferente do índice.')
                values = np.load(io.BytesIO(raw), allow_pickle=False)
            except (ValueError, EOFError, OSError) as exc:
                raise ValueError('Bloco RMS inválido '+block['member']+': '+str(exc)) from exc
            expected = tuple(b-a for a,b in zip(block['starts'],block['stops']))
            if values.shape != expected or values.dtype != self.dtype:
                raise ValueError('Conteúdo de bloco RMS diferente do índice.')
            if self.descriptor.get('finite') is True and not np.all(np.isfinite(values)):
                raise ValueError('Estado aceito RMS não finito no bloco consultado.')
            local = [selected[position]-start for selected,position,start
                     in zip(indices,positions,block['starts'])]
            output[np.ix_(*positions)] = values[np.ix_(*local)]
        return np.squeeze(output, axis=tuple(scalar_axes)) if scalar_axes else output

    def __array__(self, dtype=None, copy=None):
        if copy is False:raise ValueError('O resultado em blocos exige materialização para formar um array completo.')
        values = self[(slice(None),)*self.ndim]
        return values if dtype is None else values.astype(dtype, copy=False)

    def tobytes(self, order='C'):return np.asarray(self).tobytes(order)

    def __deepcopy__(self, memo):
        value = np.asarray(self); memo[id(self)] = value
        return value


def _copy_saved_array(archive,prefix,array,filename,block_bytes,finite):
    """Reuse immutable verified NPY blocks, with one scoped source ZIP handle.

    Other readers/legacy oversize blocks retain the established writer. The
    temporary raw bytes and decoded validation array are bounded by one block;
    no precision conversion, new format, persistent handle or data cache.
    """
    if not isinstance(array,ChunkedArray) or type(array.reader) is not ResultReader:return None
    original=copy.deepcopy(array.descriptor)
    if any(type(b.get('nbytes'))is not int or b['nbytes']>block_bytes+16384
           or type(b.get('crc32'))is not int for b in original['blocks']):return None
    checked=ChunkedArray(array.reader,original)
    if checked.shape!=array.shape or checked.dtype!=array.dtype:
        raise ValueError('Índice RMS modificado durante a gravação.')
    import time
    started=time.perf_counter();read_s=0.;decode_validate_s=0.;output_s=0.;copied_bytes=0
    reader=array.reader;blocks=[]
    with reader.lock:
        if reader.closed:raise ValueError('A consulta dos resultados foi encerrada.')
        if _identity(reader.filename)!=reader.identity:
            raise ValueError('O arquivo dos resultados mudou. Reabra o estudo antes de gravar.')
        with zipfile.ZipFile(reader.filename)as source:
            for block in original['blocks']:
                before=time.perf_counter();info=source.getinfo(block['member'])
                if info.file_size!=block['nbytes'] or info.CRC!=block['crc32']:
                    raise ValueError('Integridade do bloco diferente do índice: '+block['member'])
                raw=source.read(info)  # Includes the source ZIP CRC check.
                read_s+=time.perf_counter()-before
                before=time.perf_counter()
                shape=tuple(b-a for a,b in zip(block['starts'],block['stops']))
                header=io.BytesIO(raw);version=np.lib.format.read_magic(header)
                if version==(1,0):
                    stored_shape,_,stored_dtype=np.lib.format.read_array_header_1_0(header)
                elif version==(2,0):
                    stored_shape,_,stored_dtype=np.lib.format.read_array_header_2_0(header)
                else:
                    # NPY3 Unicode headers use NumPy's own version-aware parser.
                    # The portable NumPy2 exposes it in _format_impl; older
                    # layouts expose the same implementation in format.
                    parser=getattr(np.lib.format,'_read_array_header',None)
                    if parser is None:
                        from numpy.lib._format_impl import _read_array_header as parser
                    stored_shape,_,stored_dtype=parser(header,version)
                if (stored_shape!=shape or stored_dtype!=array.dtype or stored_dtype.hasobject
                        or len(raw)!=header.tell()+math.prod(shape)*array.dtype.itemsize):
                    raise ValueError('Cabeçalho de bloco RMS diferente do índice: '+block['member'])
                decoded=np.load(io.BytesIO(raw),allow_pickle=False)
                if not isinstance(decoded,np.ndarray) or decoded.shape!=shape or decoded.dtype!=array.dtype:
                    raise ValueError('Conteúdo de bloco RMS diferente do índice: '+block['member'])
                if finite and decoded.dtype.kind not in 'iuf':
                    raise ValueError('Estado aceito RMS inválido ao gravar o bloco.')
                if (finite or original.get('finite')is True)and not np.all(np.isfinite(decoded)):
                    raise ValueError('Estado RMS não finito no bloco: '+block['member'])
                decode_validate_s+=time.perf_counter()-before
                member=prefix+f'/values/{len(blocks):08d}.npy'
                before=time.perf_counter()
                with archive.open(member,'w',force_zip64=True)as stream:stream.write(raw)
                output_s+=time.perf_counter()-before
                target=archive.getinfo(member)
                if target.file_size!=len(raw) or target.CRC!=info.CRC:
                    raise ValueError('Integridade do bloco copiado diferente da origem.')
                blocks.append(dict(member=member,starts=list(block['starts']),stops=list(block['stops']),
                    nbytes=target.file_size,crc32=target.CRC))
                copied_bytes+=len(raw)
        if _identity(reader.filename)!=reader.identity:
            raise ValueError('O arquivo dos resultados mudou durante a cópia.')
        reader.stats['copied_numeric_members']=reader.stats.get('copied_numeric_members',0)+len(blocks)
        reader.stats['copied_numeric_bytes']=reader.stats.get('copied_numeric_bytes',0)+copied_bytes
    descriptor=dict(shape=list(array.shape),dtype=array.dtype.str,blocks=blocks)
    if finite or original.get('finite')is True:descriptor['finite']=True
    def committed():array.__init__(ResultReader(filename),descriptor)
    archive._result_commit_callbacks.append(committed)
    return descriptor


def write_array(archive, prefix, array, filename, block_bytes=ARRAY_BLOCK_BYTES, channels=16, finite=False):
    copied=_copy_saved_array(archive,prefix,array,filename,block_bytes,finite)
    if copied is not None:return copied
    shape = tuple(array.shape); dtype = np.dtype(array.dtype); blocks = []
    width = min(channels, max(1, shape[1]))
    rows = max(1, block_bytes // max(1, dtype.itemsize * width))
    for group in range(shape[2]):
        for col in range(0, shape[1], width):
            for row in range(0, shape[0], rows):
                starts = [row,col,group]; stops = [min(row+rows,shape[0]),min(col+width,shape[1]),group+1]
                member = prefix + f'/values/{len(blocks):08d}.npy'
                selected = array[tuple(slice(a,b) for a,b in zip(starts,stops))]
                if finite and (selected.dtype.kind not in 'iuf' or not np.all(np.isfinite(selected))):
                    raise ValueError('Estado aceito RMS inválido ao gravar o bloco.')
                with archive.open(member, 'w', force_zip64=True) as stream:
                    np.save(stream, selected, allow_pickle=False)
                info = archive.getinfo(member)
                blocks.append(dict(member=member, starts=starts, stops=stops,
                                   nbytes=info.file_size, crc32=info.CRC))
    descriptor = dict(shape=list(shape), dtype=dtype.str, blocks=blocks)
    if finite:descriptor['finite'] = True
    if isinstance(array, ChunkedArray):
        def committed():array.__init__(ResultReader(filename), descriptor)
        archive._result_commit_callbacks.append(committed)
    return descriptor
