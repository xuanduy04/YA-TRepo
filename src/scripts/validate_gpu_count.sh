#!/bin/bash

# Number of GPUs expected per node
N_GPUS_PER_NODE=$1

# Detect actual GPU count
ACTUAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [[ $ACTUAL_GPUS -eq 0 ]]; then
    echo "ERROR: Could not detect any GPUs. Is nvidia-smi available?" >&2
    exit 1
fi

if [[ $ACTUAL_GPUS -lt $N_GPUS_PER_NODE ]]; then
    echo "ERROR: Expected ${N_GPUS_PER_NODE} GPU(s), but only found ${ACTUAL_GPUS}." >&2
    echo "       This node does not meet the minimum GPU requirement. Aborting." >&2
    exit 1

elif [[ $ACTUAL_GPUS -gt $N_GPUS_PER_NODE ]]; then
    cat >&2 <<EOF

  ╔════════════════════════════════════════════════════════════════╗
  ║                                                                ║
  ║   ██╗    ██╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗    ║
  ║   ██║    ██║██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝    ║
  ║   ██║ █╗ ██║███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗   ║
  ║   ██║███╗██║██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║   ║
  ║   ╚███╔███╔╝██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝   ║
  ║    ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝    ║
  ║                                                                ║
  ║  !!!!!!!!!!!!!!!!!!  GPU COUNT MISMATCH  !!!!!!!!!!!!!!!!!!!!  ║
  ║                                                                ║
  ║   Expected : ${N_GPUS_PER_NODE} GPU(s)                         ║
  ║   Found    : ${ACTUAL_GPUS} GPU(s)   <── MORE THAN EXPECTED!   ║
  ║                                                                ║
  ║   This node has MORE GPUs than N_GPUS_PER_NODE specifies.      ║
  ║   Your job may only use a SUBSET of available GPUs, leaving    ║
  ║   hardware idle — or worse, interfere with other workloads.    ║
  ║                                                                ║
  ║   Double-check CUDA_VISIBLE_DEVICES / your launcher config!    ║
  ║                                                                ║
  ╚════════════════════════════════════════════════════════════════╝

EOF

else
    echo "GPU check passed: found exactly ${ACTUAL_GPUS} GPU(s) as expected."
fi
