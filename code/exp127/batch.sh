#!/bin/sh
# usage: sh batch.sh n mode val minutes  -> results/n{n}_{mode}.log
n=$1; mode=$2; val=$3; min=$4; G=$([ "$n" = 9 ] && echo 37 || echo 30)
cd "$(dirname "$0")"
for f in MMM SSS3 MMS MSS1 SSS2 MSS0 SSS1 SSS0; do
  d=6; [ "$f" = SSS0 ] && d=10; [ "$f" = SSS1 ] || [ "$f" = MSS0 ] && d=8
  python run.py $n $f $mode $val $G 14 $min $d >> results/n${n}_${mode}.log 2>&1
done
echo BATCH DONE >> results/n${n}_${mode}.log
