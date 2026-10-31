from __future__ import annotations

import socket
import struct
from pathlib import Path

import numpy as np


class ReferenceRequired(ValueError):
    """Packet structures outside the validated direct-parser contract."""


def timestamp_ns(ticks, resolution, offset):
    binary, exponent = resolution
    denominator = 2**exponent if binary else 10**exponent
    return ticks * 1_000_000_000 // denominator + offset * 1_000_000_000


def interface_options(data, begin, end, endian):
    resolution, offset = (False, 6), 0
    while begin + 4 <= end:
        code, length = struct.unpack_from(endian+'HH', data, begin)
        begin += 4
        if code == 0:
            break
        if begin + length > end:
            raise ReferenceRequired('Truncated interface option')
        if code == 9 and length == 1:
            raw = data[begin]
            resolution = (bool(raw & 128), raw & 127)
        elif code == 14 and length == 8:
            offset = struct.unpack_from(endian+'q', data, begin)[0]
        begin += (length+3) & ~3
    return resolution, offset


def packet_direction(data, begin, length, client):
    end = begin + length
    if length < 14:
        raise ReferenceRequired('Truncated Ethernet header')
    ethertype = struct.unpack_from('!H', data, begin+12)[0]
    ip = begin+14
    while ethertype in (0x8100, 0x88A8, 0x9100):
        if ip+4 > end:
            raise ReferenceRequired('Truncated VLAN header')
        ethertype = struct.unpack_from('!H', data, ip+2)[0]
        ip += 4
    if ethertype != 0x0800:
        # The reference uses this session's IPv4 client IP, so IPv6/ARP cannot match.
        return 0
    if ip+20 > end or data[ip] >> 4 != 4:
        raise ReferenceRequired('Invalid or truncated IPv4 header')
    ihl = (data[ip] & 15)*4
    if ihl < 20 or ip+ihl > end:
        raise ReferenceRequired('Invalid IPv4 header length')
    src = data[ip+12:ip+16]
    dst = data[ip+16:ip+20]
    direction = 1 if src == client else (-1 if dst == client else 0)
    if direction == 0:
        return 0
    fragment = struct.unpack_from('!H', data, ip+6)[0]
    if fragment & 0x3FFF:
        raise ReferenceRequired('IPv4 fragmentation requires reference reassembly')
    protocol = data[ip+9]
    if protocol not in (6, 17):
        return 0
    transport = ip+ihl
    if transport+4 > end:
        raise ReferenceRequired('Truncated transport header')
    source_port, target_port = struct.unpack_from('!HH', data, transport)
    # These are candidate selectors, not a replacement for all Wireshark dissectors.
    # A full-corpus reference comparison is mandatory before reporting these timings.
    if source_port == 53 or target_port == 53:
        raise ReferenceRequired('DNS dissection requires the reference backend')
    if protocol == 17 and source_port != 443 and target_port != 443:
        raise ReferenceRequired('Nonstandard-port UDP requires QUIC reference dissection')
    return direction


def read_directions(path: Path, client_ip: str, maximum: int | None = 526):
    data = path.read_bytes()
    if data[:4] != b'\x0a\x0d\x0d\x0a':
        raise ReferenceRequired('This direct backend supports PCAPNG only')
    client = socket.inet_aton(client_ip)
    position, endian, interfaces = 0, '<', []
    first_ns = None
    values, times = [], []
    while position+12 <= len(data):
        if data[position:position+4] == b'\x0a\x0d\x0d\x0a':
            bom = data[position+8:position+12]
            if bom == b'\x4d\x3c\x2b\x1a':
                endian = '<'
            elif bom == b'\x1a\x2b\x3c\x4d':
                endian = '>'
            else:
                raise ReferenceRequired('Invalid section byte order')
        kind, block_length = struct.unpack_from(endian+'II', data, position)
        if block_length < 12 or block_length % 4 or position+block_length > len(data):
            raise ReferenceRequired('Invalid PCAPNG block length')
        if struct.unpack_from(endian+'I', data, position+block_length-4)[0] != block_length:
            raise ReferenceRequired('PCAPNG trailing block length mismatch')
        body = position+8
        if kind == 0x0A0D0D0A:
            interfaces = []
        elif kind == 1:
            if block_length < 20:
                raise ReferenceRequired('Invalid interface block')
            link_type = struct.unpack_from(endian+'H', data, body)[0]
            resolution, offset = interface_options(data, body+8, position+block_length-4, endian)
            interfaces.append((link_type, resolution, offset))
        elif kind == 6:
            if block_length < 32:
                raise ReferenceRequired('Invalid enhanced packet block')
            interface, high, low, captured, original = struct.unpack_from(endian+'IIIII', data, body)
            if interface >= len(interfaces) or body+20+captured > position+block_length-4:
                raise ReferenceRequired('Invalid packet block payload')
            link_type, resolution, offset = interfaces[interface]
            if link_type != 1:
                raise ReferenceRequired('Non-Ethernet capture')
            ns = timestamp_ns((high << 32)+low, resolution, offset)
            if first_ns is None:
                first_ns = ns
            elapsed_ns = ns-first_ns
            if elapsed_ns < 0:
                raise ReferenceRequired('Out-of-order capture timestamps')
            if elapsed_ns <= 20_000_000_000:
                direction = packet_direction(data, body+20, captured, client)
                if direction:
                    values.append(direction)
                    times.append(elapsed_ns/1_000_000_000)
                    if maximum is not None and len(values) >= maximum:
                        break
        elif kind in (2, 3):
            raise ReferenceRequired('Legacy/simple packet block requires reference')
        position += block_length
    return np.asarray(values, dtype=np.int8), np.asarray(times, dtype=np.float64)


def post506(values):
    output = np.zeros((1, 1, 506), dtype=np.float32)
    segment = values[20:526]
    output[0, 0, :len(segment)] = segment
    return output
