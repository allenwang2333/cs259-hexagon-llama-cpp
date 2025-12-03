#!/bin/sh
#

# Basedir on device
basedir=/data/local/tmp/llama.cpp

cli_opts=

branch=.
[ "$B" != "" ] && branch=$B

adbserial=
[ "$S" != "" ] && adbserial="-s $S"

model="Qwen3-VL-2B-Instruct-q4_0-LORA.gguf"
[ "$M" != "" ] && model="$M"

mmproj="mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
[ "$M" != "" ] && mmproj="$MM"

device="none"
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

# --- new: parse command-line args for image and prompt ---
while [ "$1" != "" ]; do
  case $1 in
    --image )  shift; image=$1 ;;
    --prompt ) shift; prompt=$1 ;;
    * ) cli_opts="$cli_opts $1" ;;
  esac
  shift
done

set -x



adb $adbserial shell " \
  cd $basedir; ulimit -c unlimited;        \
    LD_LIBRARY_PATH=$basedir/$branch/lib   \
    ADSP_LIBRARY_PATH=$basedir/$branch/lib \
    $experimental $sched $opmask $profile $nhvx $ndev           \
      ./$branch/bin/llama-mtmd-cli --list-devices
"
