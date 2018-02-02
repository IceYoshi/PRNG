#!/bin/sh

calc_avg()
{
	avg=`cat "$1" | lib/calc_average.awk`
	name=`echo "$1" | sed "s|${TEST_PREFIX}||"`
	echo "\"$name\"" $avg
}

echo "# name avg_runtime"

for i in $*; do
	calc_avg $i
done | sort
