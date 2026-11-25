# Running on AWS SageMaker

A SageMaker training job runs one command in this image on one GPU instance and stops by itself. Nothing here is
account-specific; the account's bucket, image repository, role and job requests are kept outside the repository.

| File | What |
|---|---|
| `Dockerfile` (+ `Dockerfile.dockerignore`) | The image: both environments from `scripts/setup_envs.sh`; unit tests and the frozen-module check run during the build. Build from the repository root: `DOCKER_BUILDKIT=1 docker build -f aws/Dockerfile .` |
| `entrypoint.sh` | Writes the runtime `.env` (data on the job's storage volume), links `outputs/` to a work directory on the job's volume (not `/opt/ml/checkpoints`: SageMaker's upload files there break code that lists a directory it has just written; the job script mirrors metrics, logs and records there every 2 minutes, and the trained task's weights at exit), runs the given command |
| `clip3_seq_job.sh` | The clipped curriculum (`scripts/cl/clip3_sequence.sh`, `configs/cl_opd_clip3.yaml`: tasks 1-4 with per-trajectory clipping k=3, 400 updates each, from the unclipped task 0 of `outputs/cl_opd/full_20261108T214604Z`). Built for managed spot and for stopping at any time: `scripts/cl/persist.py` copies the run (all files except diagnostic batches) and every task's full training state (master, optimizer, replay, sampler; two alternating checksummed slots, refreshed every 10 updates) to `/opt/ml/checkpoints/cl_opd/` every 2 minutes and after every stage; on SIGTERM the trainer saves its state and a last snapshot is taken. A job with the same checkpoint S3 URI (a spot restart, or a new job with the same run id) restores it and continues. Exit code 6 = the restore failed |
| `clip_pilot_job.sh` | A t063 pilot, picked by its argument: `clip` (default; `scripts/cl/t063_clip_pilot.sh`), `tr` (the trust-region arm, `configs/cl_opd_t063_tr3_u200.yaml`) or `continue` (the clipping pilot from its update-190 state to 400, `scripts/cl/t063_clip_continue.sh`; input channel `pilot` = the pilot run's `k1_t063/`): checks GPU and NVIDIA EGL rendering, downloads its public inputs (OpenPI base checkpoint; teachers and pilot inputs from the Hugging Face repo, sha256-checked), runs it, writes `/opt/ml/model/results.tar.gz` (metrics, logs, evaluation records; no weights) and `monitor.csv` (host RAM and GPU memory every 30 s) |

Job command: `/opt/vla/aws/entrypoint.sh bash aws/clip_pilot_job.sh [clip|tr|continue]`. Instance: `ml.g6e.8xlarge` (1 L40S 48 GB, 256 GiB host
RAM: the pathwise-gradient compile needs about 120 GiB). Use on-demand for a first run; set `MaxRuntimeInSeconds` as
the cost cap. Exit codes of the job script: 4 = no NVIDIA EGL rendering, 5 = an input download or checksum failed,
otherwise the pilot's own.

Job command for the curriculum: `/opt/vla/aws/entrypoint.sh bash aws/clip3_seq_job.sh` (spot; `MaxRuntimeInSeconds` is the
cost cap, and the job can be continued later). To continue elsewhere, copy the checkpoint prefix of the run id (its
`cl_opd/` directory) to a directory DEST, then `python3 scripts/cl/persist.py restore --from DEST --outputs outputs` and
`scripts/cl/clip3_sequence.sh`; `scripts/cl/persist.py status --from DEST` lists what each task has (finished, latest
saved update, checkpoints).
