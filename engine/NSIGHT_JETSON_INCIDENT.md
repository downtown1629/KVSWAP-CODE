# Jetson Nsight Hardware-Counter Incident

## Summary

On 2026-08-13, hardware-counter profiling on the Orin Nano caused the local
tty1 display to stop updating. SSH and the rest of the operating system stayed
responsive. No OOM, GPU XID, GPU timeout, kernel panic, or residual profiler
process was observed. Treat Nsight hardware-counter collection as unsafe on
this software image until the NVIDIA tool/driver compatibility is resolved.

## Affected environment

- Jetson Orin Nano, 8 GiB unified memory
- L4T R36.5.2, Linux `5.15.199-tegra`
- Nsight Systems `2024.5.4.34`
- Nsight Compute `2024.3.1`
- Console session on tty1; system default target is `multi-user.target`

## Trigger and evidence

A normal non-root `nsys --trace=cuda,nvtx` CUDA smoke completed successfully.
The failure followed root hardware-counter probing with:

```text
nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=ga10b ...
ncu --set basic --target-processes all ...
```

Kernel errors began immediately after the first hardware-metric command and
intensified when `ncu` started. Representative messages were:

```text
tegra_hwpm_aperture_for_address: Address ... not in any IP
tegra_hwpm_exec_regops: exec_reg_ops 0 failed
nvgpu_regops_exec: invalid op(s)
nvgpu_prof_ioctl_exec_reg_ops: regop execution failed
validate_reg_op_offset: invalid regop offset
```

The errors stopped after the profiler was terminated. The active VT remained
tty1, framebuffer mode remained 3840x2160, and the login shell stayed alive,
but physical display output did not recover automatically.

## Profiling policy

Do not run either of the following on this Jetson image:

- `ncu` / Nsight Compute kernel replay or performance-counter collection.
- `nsys` with `--gpu-metrics-devices`, `--gpu-metrics-set`, Tegra accelerator
  counters, or any other HWPM/performance-monitor configuration.

Permitted first-line profiling is software tracing only:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none ...
```

Keep captures short and restricted to the engine's existing
`cudaProfilerStart/Stop` range. Continue using `jtop_logger.py` for GPU/EMC,
power, memory, and NVMe sampling. Do not escalate to root merely to expose
additional counters.

## Recovery and diagnostics

Before changing system state, confirm that no `ncu`, `nsys`, or profiled CUDA
child remains and inspect `journalctl -k` for HWPM/nvgpu errors. For a console
that remains blank while the system is healthy, the least invasive recovery is
a VT switch (`Ctrl+Alt+F2`, then `Ctrl+Alt+F1`, or equivalent `chvt` commands).
Reboot or display-service changes require explicit operator approval.

Preserve the incident evidence with:

```bash
sudo journalctl -k --since "2026-08-13 16:11:00" \
  | grep -Ei 'nvgpu|hwpm|regop|gpu|drm|host1x'
```
