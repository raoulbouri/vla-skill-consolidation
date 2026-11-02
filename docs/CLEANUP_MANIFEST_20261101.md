# Cleanup manifest, 2026-11-01

**Deleted: 51 files, about 19.2 GiB reclaimed** (42 files on xulab, 9 on aims). Nothing was committed or pushed. Deletion used an explicit file list, with no wildcards or recursive removal.

Duplicate gradient dumps were compared array by array (dtype, shape, bytes) first; the record is `outputs/ogpo_port/task10b/duplicate_check.json`. `p2_default_2.npz` differed from `_1`, so it was kept.

Rerun note: `task10b_compare.py` and `task10c_inventory.py` read the deleted `p3_b_*` files. Rerunning them needs those repeats regenerated with `task10b.py`. Their saved outputs (`task10b/analysis.json`, `task10c/inventory.json`) are retained.

## Deleted on xulab (`/mnt/data/users/bbouri/IRL-project`)

| Path | Class | Why safe | Bytes | sha256 prefix |
|---|---|---|---|---|
| `outputs/ogpo_port/task10b/p2_det_2.npz` | duplicate output | All 24 arrays identical to retained p2_det_1.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `493895c58be4fbe4` |
| `outputs/ogpo_port/task10b/p2_det_3.npz` | duplicate output | All 24 arrays identical to retained p2_det_1.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `493895c58be4fbe4` |
| `outputs/ogpo_port/task10b/p2_det_4.npz` | duplicate output | All 24 arrays identical to retained p2_det_1.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `493895c58be4fbe4` |
| `outputs/ogpo_port/task10b/p2_nt_2.npz` | duplicate output | All 24 arrays identical to retained p2_nt_1.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `37b88582791f4002` |
| `outputs/ogpo_port/task10b/p3_b_11000.npz` | duplicate output | All 24 arrays identical to retained p3_a_11000.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `58f0d2e3f7b48c1a` |
| `outputs/ogpo_port/task10b/p3_b_11001.npz` | duplicate output | All 24 arrays identical to retained p3_a_11001.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `d5837f7aceef028e` |
| `outputs/ogpo_port/task10b/p3_b_11002.npz` | duplicate output | All 24 arrays identical to retained p3_a_11002.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `7f893d3609d568a3` |
| `outputs/ogpo_port/task10b/p3_b_11003.npz` | duplicate output | All 24 arrays identical to retained p3_a_11003.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `9828f4249e3710b7` |
| `outputs/ogpo_port/task10b/p3_b_11004.npz` | duplicate output | All 24 arrays identical to retained p3_a_11004.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `d3a7f411d6eb5446` |
| `outputs/ogpo_port/task10b/p3_b_11005.npz` | duplicate output | All 24 arrays identical to retained p3_a_11005.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `a788b1ff3b1d84c5` |
| `outputs/ogpo_port/task10b/p3_b_11006.npz` | duplicate output | All 24 arrays identical to retained p3_a_11006.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `733eb655698c87f8` |
| `outputs/ogpo_port/task10b/p3_b_11007.npz` | duplicate output | All 24 arrays identical to retained p3_a_11007.npz (dtype, shape, bytes); sidecar JSON kept | 1720440628 | `30ba74a79c92480b` |
| `logs/task11_real_20261101T205005Z.log` | temporary log | Crashed on a script bug; superseded by task11_real_20261101T205157Z.log | 4591 | `c95be97a0d24371a` |
| `logs/task11_synthetic_20261101T204605Z.log` | temporary log | Crashed printing checks; superseded by task11_synthetic_20261101T205157Z.log, ALL_PASS | 4771 | `57b1bb324d18e3e2` |
| `logs/task11_synthetic_20261101T204754Z.log` | temporary log | Byte-identical to the previous crash; superseded the same way | 4771 | `57b1bb324d18e3e2` |
| `logs/task10d_fd_20261101T155025Z.log` | temporary log | Failed launch on a missing obs 1 cache; no result; real run is task10d_fd_20261101T155250Z.log, which the ledger cites | 1076 | `e37706de872bec67` |
| `logs/task10a_derivatives_fp32_det_20261031T231854Z.log` | temporary log | Same 15.4 GB allocation failure as retained 230714Z log | 1289 | `d053dc372b54bcfc` |
| `logs/task10a_derivatives_fp32_det_20261031T232143Z.log` | temporary log | Same 15.4 GB allocation failure as retained 230714Z log | 1289 | `02ac19937d1798f8` |
| `logs/task10b_dot_20261101T032603Z.log` | temporary log | Byte-identical to retained 032601Z log | 1815 | `1e4cefc4fefa4f79` |
| `logs/cleanup_dupcheck.log` | temporary log | This cleanup's own check output; results kept in task10b/duplicate_check.json | 615 | `4146774cf945fc97` |
| `tmp/task11_dupcheck.py` | scratch file | This cleanup's own one-off check script | 927 | `7d0b1dfe6755000a` |
| `tests/ogpo_port/__pycache__/pi05_real_env.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 5747 | `526415528097c376` |
| `tests/ogpo_port/__pycache__/task10a_derivatives.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 31077 | `17a1d10918d3e274` |
| `tests/ogpo_port/__pycache__/task10a_primal.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 33292 | `4730bc429e3a5259` |
| `tests/ogpo_port/__pycache__/task10b.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 44676 | `ade299ea49e81344` |
| `tests/ogpo_port/__pycache__/task10c_compare.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 19279 | `4394a64ba3460a4d` |
| `tests/ogpo_port/__pycache__/task10d.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 59271 | `c760dc0fbb809530` |
| `tests/ogpo_port/__pycache__/test_euler_convention.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 30728 | `debf49902776ba4e` |
| `tests/ogpo_port/__pycache__/test_euler_convention.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 22228 | `4d903aa013be7ecd` |
| `tests/ogpo_port/__pycache__/test_flow_log_prob.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 56956 | `4cd3fbb70933548e` |
| `tests/ogpo_port/__pycache__/test_gaussian_path_kl.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 42138 | `77750064b31310ad` |
| `tests/ogpo_port/__pycache__/test_pathwise_kl_gradient.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 105281 | `af24c4874b424d0b` |
| `tests/ogpo_port/__pycache__/test_pathwise_kl_gradient.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 45877 | `a6b35404054b7806` |
| `tests/ogpo_port/__pycache__/test_pi05_path_kl.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 67523 | `c7e28f8e58e8efa7` |
| `tests/ogpo_port/__pycache__/test_pi05_path_kl.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 38538 | `9d5bae9db313228b` |
| `tests/ogpo_port/__pycache__/test_pi05_path_logprob.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 45112 | `01e93f8b36bb1296` |
| `tests/ogpo_port/__pycache__/test_pi05_pathwise_gradient.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 73516 | `b9ec412bcdc1984c` |
| `tests/ogpo_port/__pycache__/test_pi05_pathwise_gradient.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 46691 | `9ae5e58a59a211ca` |
| `tests/ogpo_port/__pycache__/test_pi05_sde_sampler.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 44007 | `64060ed8d9cb1180` |
| `tests/ogpo_port/__pycache__/test_pi05_velocity_adapter.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 55670 | `e1fba01c8c7d37cf` |
| `tests/ogpo_port/__pycache__/test_sde_transition.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 39600 | `a079ad21ba767a15` |
| `tests/ogpo_port/__pycache__/test_sde_transition.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 27902 | `ea690cba84ee509a` |

## Deleted on aims (`~/IRL-project`)

| Path | Class | Why safe | Bytes |
|---|---|---|---|
| `tests/ogpo_port/__pycache__/pi05_real_env.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 5742 |
| `tests/ogpo_port/__pycache__/test_euler_convention.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 30723 |
| `tests/ogpo_port/__pycache__/test_euler_convention.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 22223 |
| `tests/ogpo_port/__pycache__/test_flow_log_prob.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 56951 |
| `tests/ogpo_port/__pycache__/test_gaussian_path_kl.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 42133 |
| `tests/ogpo_port/__pycache__/test_pathwise_kl_gradient.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 59123 |
| `tests/ogpo_port/__pycache__/test_pi05_path_kl.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 67518 |
| `tests/ogpo_port/__pycache__/test_sde_transition.cpython-311-pytest-8.3.5.pyc` | cache | Python bytecode, regenerated on import | 39595 |
| `tests/ogpo_port/__pycache__/test_sde_transition.cpython-311.pyc` | cache | Python bytecode, regenerated on import | 27897 |

## Deliberately preserved as ambiguous

| Path | Size | Why kept |
|---|---|---|
| `outputs/ogpo_port/task10b/p3_a_11003..11007.npz` | 8.0 GiB | Unique gradients; not duplicated elsewhere |
| `outputs/ogpo_port/task10b/p3_a_11000..11007_eager.npz` | 12.8 GiB | Unique eager gradients behind the jit/eager comparison |
| `outputs/ogpo_port/task10b/p2_default_1.npz`, `p2_default_2.npz` | 3.2 GiB | Differ from each other: raw evidence of cross-process nondeterminism |
| `outputs/ogpo_port/task10b/p2_nt_1.npz` | 1.6 GiB | Only copy of the Triton-off gradient |
| `outputs/ogpo_port/task10c/reference_gradients/C4A_11000..11002.npz` | 9.6 GiB | Only stored FP32 references for these seeds; not referenced by later scripts |
| One-off scripts such as `task10a_probe.py`, `task10a_scan_probe.py`, `task10a_analyze.py`, `task10b_dot.py`, `task10b_jump.py`, `task10b_fdaudit.py` | Small | Unreferenced by name but they document how historic results were made |
| `tmp/` diagnostics and `tmp/memprobe/` | 160 KiB | Predate Task 10; `docs/OGPO_AUDIT.md` and `docs/SETUP_REPORT.md` cite some of them |
| `src/clvla/ogpo/__pycache__/` | Small | Left alone so nothing is touched in the frozen module directory |
| Remaining Task 10 and 11 logs | Under 1 MiB | Small, and the ledger cites them by name or pattern |
