# Running on AWS SageMaker

A SageMaker training job runs one command in this image on one GPU instance and stops by itself. Nothing here is
account-specific; the account's bucket, image repository, role and job requests are kept outside the repository.

| File | What |
|---|---|
| `Dockerfile` (+ `Dockerfile.dockerignore`) | The image: both environments from `scripts/setup_envs.sh`; unit tests and the frozen-module check run during the build. Build from the repository root: `DOCKER_BUILDKIT=1 docker build -f aws/Dockerfile .` |
| `entrypoint.sh` | Writes the runtime `.env` (data on the job's storage volume), links `outputs/` to `/opt/ml/checkpoints` (synced to S3 while the job runs), runs the given command |
| `clip_pilot_job.sh` | The t063 clipping pilot (`scripts/cl/t063_clip_pilot.sh`): checks GPU and NVIDIA EGL rendering, downloads its public inputs (OpenPI base checkpoint; teachers and pilot inputs from the Hugging Face repo, sha256-checked), runs the pilot, writes `/opt/ml/model/results.tar.gz` (metrics, logs, evaluation records; no weights) and `monitor.csv` (host RAM and GPU memory every 30 s) |

Job command: `/opt/vla/aws/entrypoint.sh bash aws/clip_pilot_job.sh`. Instance: `ml.g6e.8xlarge` (1 L40S 48 GB, 256 GiB host
RAM: the pathwise-gradient compile needs about 120 GiB). Use on-demand for a first run; set `MaxRuntimeInSeconds` as
the cost cap. Exit codes of the job script: 4 = no NVIDIA EGL rendering, 5 = an input download or checksum failed,
otherwise the pilot's own.
