#!/bin/sh


for i in $*; do
	sed -n 's/^# using dev [0-9]\+: \(.*\)/\1/p' "$i"
done | sort | uniq
