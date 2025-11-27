#!/bin/sh

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

# --- CHANGE 1: Update Default Model to Qwen ---
model="qwen3-vl-2b-instruct-q4_k_m.gguf"
[ "$M" != "" ] && model="$M"

mmproj="mmproj-model-f16.gguf"
[ "$MM" != "" ] && mmproj="$MM"
# ---------------------------------------------

device="HTP0"
[ "$D" != "" ] && device="$D"

verbose=
[ "$V" != "" ] && verbose="GGML_HEXAGON_VERBOSE=$V"

experimental=
[ "$E" != "" ] && experimental="GGML_HEXAGON_EXPERIMENTAL=$E"

sched=
[ "$SCHED" != "" ] && sched="GGML_SCHED_DEBUG=2" cli_opts="$cli_opts -v"

profile=
[ "$PROF" != "" ] && profile="GGML_HEXAGON_PROFILE=$PROF GGML_HEXAGON_OPSYNC=1"

opmask=
[ "$OPMASK" != "" ] && opmask="GGML_HEXAGON_OPMASK=$OPMASK"

nhvx=
[ "$NHVX" != "" ] && nhvx="GGML_HEXAGON_NHVX=$NHVX"

ndev=
[ "$NDEV" != "" ] && ndev="GGML_HEXAGON_NDEV=$NDEV"

# --- Parse command-line args for image and prompt ---
while [ "$1" != "" ]; do
  case $1 in
    --image )  shift; image=$1 ;;
    --prompt ) shift; prompt=$1 ;;
    * ) cli_opts="$cli_opts $1" ;;
  esac
  shift
done

# --- CHANGE 2: Qwen Specific Flags ---
# 1. -ctk q8_0 -ctv q8_0 : Use 8-bit cache for accuracy (Q4 breaks OCR)
# 2. -n 128              : Allow longer answers for OCR text
# 3. -fa                 : Flash Attention is mandatory for Qwen
# 4. Removed --chat-template: handled in Python
# -------------------------------------

adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $experimental $sched $opmask $profile $nhvx $ndev            \
      ./$branch/bin/llama-mtmd-cli -m $basedir/../gguf/$model        \
         --mmproj $basedir/../gguf/$mmproj --no-mmproj-offload     \
         --batch-size 1 -n 128 --no-mmap \
         -fa \
         -ctk q8_0 -ctv q8_0 \
         --device $device --temp 0.1 \
         --image \"$image\" -p \"$prompt\" \
"