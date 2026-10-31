# Single-Session Processing Time

## Reported Measurements

- Batch size1, one PCAPNG session input per call, not batch-throughput latency.
- Same representative input membership across models/seeds, saved in `data/metadata/latency_sample_assignments.csv`.
- All five final learning seeds and all five outer folds; 50 GPU warmup calls per checkpoint/role using a fit input, followed by three repetitions per selected session.
- For the group-aware TCN/ResNet1D test comparison, 220 unique sessions x5seeds x3repetitions =3,300 timed calls per model.
- Read file and parse through the first526 eligible packets, encode directions, optionally compute light16 and apply the fit-only scaler, transfer CPU->GPU, model forward, sigmoid/probability CPU transfer, threshold decision, final CUDA synchronization.
- Synchronize CUDA before starting the wall-clock timer (outside measured duration). Final synchronization is inside the timer. ROCKET keeps kernels on GPU and its scaler/classifier on CPU.
- `Preprocess_ms`: file read + parse + direction/feature packing. `Inference_ms`: H2D + forward + D2H + decision + finalsync. `Total_ms`: uninterrupted end-to-end wall clock. Raw boundaries and component times are public.
- Loading checkpoints, training, packet collection wait, validation assertions and output-file writes are excluded. Model/OS caches are warm; no filesystem cache flush. No concurrent training during the original measurements.
- Average three repeats for each session/checkpoint first. The main reported mean and sample SD are across these session/checkpoint means. This is **not seed SD**, **not raw-call SD**, and the three repeats are not treated as independent sessions. Seed SD and p95 are additional columns in `latency_summary.csv`.

Preprocessing is recomputed and timed within every model call, so its observed mean need not be identical across models despite identical parsing logic and input membership. OS scheduling, cache state and elapsed measurement conditions vary. It is not a model-specific preprocessing algorithm. Total SD is calculated from total observations, not by summing component SDs.

Table4(c)'s selection ROC-AUC comes from inner pilot validation. Its timing comes from the selected final checkpoints on reserved validation inputs; it is not the cost of tuning or an independent model-selection score. Figure and table boundaries should use the same continuous end-to-end definition; see the manuscript correction note.

## Public Reaggregation

```console
python -m darkmirror_repro reproduce --out artifacts/reproduced
```

This recalculates the existing measured times; it does not pretend to rerun PCAP preprocessing. Raw PCAPs and client-IP metadata are withheld, so **full independent remeasurement of the paper's PCAP-based latency is not possible from public processed data alone**. Model learning, prediction statistics and timing-record reaggregation remain public.

## Optional Private-Capture Remeasurement

After local retraining/evaluation, supply a private CSV outside the tracked repository with columns `index,source_path,client_ip`. This is a user-supplied input, not a released artifact. Never commit it. It must cover all required timing sample indices and match the public526 prefixes. TShark must be installed.

```console
python -m darkmirror_repro benchmark --out artifacts/rerun --private-manifest PATH_TO_PRIVATE_CSV --tshark PATH_TO_TSHARK
```

The tool validates all supplied captures against TShark and the released numeric input before timing. It writes only anonymous index/group measurement records to ignored `artifacts/`. Hardware and runtime are recorded anew; new timings are not expected to equal the original RTX3060Ti values. Packet collection time can exceed reported processing latency and must be added separately when discussing online detection.
