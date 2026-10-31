"""Reference extraction for externally supplied private captures."""
from pathlib import Path
import subprocess

import numpy as np

DISPLAY_FILTER = '(tcp or tls or quic) and not (dns or icmp or icmpv6)'


def reference_directions(path, client_ip, maximum=None, tshark='tshark'):
    command = [str(tshark), '-n', '-r', str(path), '-Y', DISPLAY_FILTER, '-T', 'fields',
               '-E', 'separator=\t', '-E', 'occurrence=f']
    for field in ['frame.number', 'frame.time_relative', 'ip.src', 'ip.dst', 'ip.proto', 'frame.protocols']:
        command += ['-e', field]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='strict',
                            timeout=180, check=True)
    directions, times = [], []
    for line in result.stdout.splitlines():
        fields = (line.split('\t') + [''] * 6)[:6]
        elapsed = float(fields[1])
        if elapsed > 20.:
            continue
        if fields[2] == client_ip:
            direction = 1
        elif fields[3] == client_ip:
            direction = -1
        else:
            continue
        layers = fields[5].split(':')
        if any(layer in layers for layer in ['dns', 'icmp', 'icmpv6']):
            raise ValueError('Unexpected excluded protocol')
        if not (fields[4] == '6' or (fields[4] == '17' and 'quic' in layers)):
            raise ValueError('Unvalidated transport protocol')
        directions.append(direction)
        times.append(elapsed)
        if maximum is not None and len(directions) >= maximum:
            break
    if not directions:
        raise ValueError('No eligible packets; do not silently exclude this session.')
    return np.asarray(directions, dtype=np.int8), np.asarray(times, dtype=np.float64)


def encode(direction, length=526):
    direction = np.asarray(direction, dtype=np.int8)
    if direction.ndim != 1 or not np.isin(direction, [-1, 1]).all():
        raise ValueError('Only known client/server directions may be encoded.')
    result = np.zeros(length, dtype=np.int8)
    n = min(len(direction), length)
    result[:n] = direction[:n]
    return result


def validate_direct(manifest, root, tshark='tshark'):
    """Private manifest is never written to a public output."""
    from .direct_parser import read_directions
    from .io import dataset
    data = dataset(root)
    for row in manifest.itertuples():
        direction, times = read_directions(Path(row.source_path), row.client_ip, 526)
        reference, reference_times = reference_directions(row.source_path, row.client_ip, 526, tshark)
        assert np.array_equal(direction, reference)
        assert np.allclose(times, reference_times, atol=1e-8, rtol=0)
        assert np.array_equal(direction, data['prefix'][row.index, :526])
        assert np.allclose(times, data['relative_times_s'][row.index, :526], atol=1e-8, rtol=0)
