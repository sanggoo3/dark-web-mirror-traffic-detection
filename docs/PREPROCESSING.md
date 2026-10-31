# Preprocessing Contract

Raw captures are private. The published sequences are the **post-preprocessing learning data** and are sufficient for the reported model experiments. Reference extraction code is included to document the procedure and support separately authorized private captures.

1. Use capture order and `frame.time_relative`, relative to the first captured frame. Keep eligible packets at elapsed time <=20 seconds.
2. Reference TShark display filter: `(tcp or tls or quic) and not (dns or icmp or icmpv6)`.
3. Verify the outer IPv4 transport is TCP, or UDP with an actual QUIC layer. A quoted TCP header inside ICMP must not make an ICMP packet eligible. This correction distinguishes the final data from historical caches.
4. With the known session client IPv4 address, encode client source +1, client destination -1. Discard unmatched direction; do not encode it as0. Keep capture order and the corresponding relative timestamps.
5. Save the first1276 directions for the fixed-length comparisons. For any input length L, truncate at L or right-pad with zero; matching timestamp padding is NaN, never a genuine timing observation. Zero is not a packet direction.
6. Record actual <=1/3/5/10-second eligible counts and the full <=20-second eligible count separately. These counts do not time-crop the classification inputs.
7. Primary input uses `q[:526]`, initial20 retained. Post-handshake control uses `q[20:526]`, exactly506. Do not discard20 from an already506-length cache or refill it with packets after526.

`preprocessing.reference_directions` implements the reference protocol dissection; `encode` implements truncation/right padding. `direct_parser.read_directions` implements the corpus-validated PCAPNG fast path used for timing. The original files have `.pcap` names but PCAPNG contents.

## Direct Parser Limit

The fast parser supports validated Ethernet/IPv4 PCAPNG records and uses transport-port candidates. It does **not** implement general Wireshark QUIC dissection/reassembly. Unsupported link types, fragmentation, DNS-port traffic or nonstandard UDP require the reference backend. A UDP443 selector is not universally equivalent to a QUIC filter. Before reporting timings on any new capture set, `validate_direct` compares every supplied capture against the reference extractor and released input prefix. Unsupported/mismatching inputs fail rather than silently changing eligibility. Do not claim general protocol equivalence from this corpus-specific acceleration.

No collection scripts, endpoint addresses, private source manifests or capture payloads are distributed. The exact dataset is identified by numeric archive/file hashes, not original filenames. The pseudonymous split assignments are already frozen; no URL lookup or network access is needed for learning or result aggregation.
