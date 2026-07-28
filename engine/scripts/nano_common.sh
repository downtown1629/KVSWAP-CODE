#!/bin/bash
# Shared helpers for the Jetson Orin Nano (8 GB RAM, NVMe-only) variant of the
# engine/ scripts: setup_nano.sh, download_models_nano.sh, eval_nano.sh.
# Meant to be `source`d, not run directly.
#
# scripts/setup.sh and scripts/eval.sh hard-require "AGX Orin" in the
# device-tree model string and an exact nvpmodel mode of "MAXN", and probe a
# GPU devfreq sysfs path that was only verified on AGX Orin. Those checks
# have not been verified on Orin Nano, so here they're advisory: we print
# what we find and let you decide, instead of exiting. Set STRICT_HW_CHECK=1
# to make the hardware-model check fatal again.

check_hardware_soft() {
  if [[ ! -r /proc/device-tree/model ]]; then
    echo "[hw-check] WARNING: cannot read /proc/device-tree/model (not a Jetson? continuing anyway)"
    return 0
  fi
  local model
  model="$(tr -d '\0' < /proc/device-tree/model)"
  echo "[hw-check] Detected device-tree model: ${model}"
  case "$model" in
    *"Orin Nano"*)
      echo "[hw-check] Orin Nano detected, OK"
      ;;
    *"Orin"*)
      echo "[hw-check] NOTE: this is '${model}', not an Orin Nano. The *_nano.sh scripts were written for 8GB Orin Nano + NVMe-only, but should still work on any Orin SKU with more headroom."
      ;;
    *)
      echo "[hw-check] WARNING: '${model}' is not a recognized Orin device."
      if [[ "${STRICT_HW_CHECK:-0}" == "1" ]]; then
        echo "[hw-check] STRICT_HW_CHECK=1: exiting."
        exit 1
      fi
      ;;
  esac
}

check_powermode_soft() {
  if ! command -v nvpmodel >/dev/null 2>&1; then
    echo "[hw-check] nvpmodel not found; skipping power-mode check"
    return 0
  fi
  local current
  current="$(nvpmodel -q --verbose 2>/dev/null | awk -F'NV Power Mode: ' '/NV Power Mode:/{print $2; exit}' | tr -d '\r' | xargs)"
  echo "[hw-check] nvpmodel reports: '${current:-unknown}'"
  case "$current" in
    *MAXN*)
      echo "[hw-check] max-performance mode, OK"
      ;;
    *)
      echo "[hw-check] WARNING: doesn't look like a max-performance mode. For throughput numbers you trust, run: sudo nvpmodel -m 0 && sudo jetson_clocks (confirm the max-perf mode id/name with 'nvpmodel -p --verbose' first — it varies by Nano carrier board/JetPack version)."
      if [[ "${STRICT_HW_CHECK:-0}" == "1" ]]; then
        echo "[hw-check] STRICT_HW_CHECK=1: exiting."
        exit 1
      fi
      ;;
  esac
}

check_jetson_clocks_soft() {
  local cpu_policy="/sys/devices/system/cpu/cpufreq/policy0"
  if [[ -f "${cpu_policy}/scaling_min_freq" && -f "${cpu_policy}/scaling_max_freq" && -f "${cpu_policy}/scaling_available_frequencies" ]]; then
    local cpu_min cpu_max cpu_avail_max
    cpu_min="$(cat "${cpu_policy}/scaling_min_freq" 2>/dev/null)"
    cpu_max="$(cat "${cpu_policy}/scaling_max_freq" 2>/dev/null)"
    cpu_avail_max="$(awk '{for(i=1;i<=NF;i++){v=$i+0; if(v>max) max=v}} END{print max+0}' "${cpu_policy}/scaling_available_frequencies" 2>/dev/null)"
    if [[ -n "$cpu_avail_max" && ( "$cpu_min" != "$cpu_avail_max" || "$cpu_max" != "$cpu_avail_max" ) ]]; then
      echo "[hw-check] WARNING: CPU not pinned to max (min=${cpu_min} max=${cpu_max} avail_max=${cpu_avail_max}). Run: sudo jetson_clocks"
    else
      echo "[hw-check] CPU clocks pinned to max, OK"
    fi
  else
    echo "[hw-check] Could not read ${cpu_policy}/*; skipping CPU clock check"
  fi

  # GPU devfreq sysfs location varies across Orin SKUs/JetPack versions; try a
  # few known candidates instead of hard-failing like scripts/setup.sh does.
  local gpu_dev=""
  for cand in /sys/devices/platform/17000000.gpu/devfreq_dev /sys/class/devfreq/17000000.gpu.devfreq /sys/class/devfreq/*.gpu*; do
    if [[ -d "$cand" ]]; then gpu_dev="$cand"; break; fi
  done
  if [[ -n "$gpu_dev" && -f "${gpu_dev}/min_freq" && -f "${gpu_dev}/max_freq" ]]; then
    local gpu_min gpu_max
    gpu_min="$(cat "${gpu_dev}/min_freq" 2>/dev/null)"
    gpu_max="$(cat "${gpu_dev}/max_freq" 2>/dev/null)"
    if [[ "$gpu_min" == "$gpu_max" ]]; then
      echo "[hw-check] GPU devfreq @ ${gpu_dev}: pinned to ${gpu_max}, OK"
    else
      echo "[hw-check] WARNING: GPU devfreq @ ${gpu_dev}: min=${gpu_min} max=${gpu_max} (not pinned). Run: sudo jetson_clocks"
    fi
  else
    echo "[hw-check] Could not locate a GPU devfreq node; skipping GPU clock check. Run 'sudo jetson_clocks' manually and spot-check with tegrastats."
  fi
}
