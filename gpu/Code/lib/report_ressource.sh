#!/bin/bash

echo "rbg_name registers lmem cmem smem"

for i in $*; do
	rng_name="$i"
	registers=`sed -n 's/.*Used \([0-9]\+\) registers.*/\1/p' "$BUILD_LOGS/rnd_to_stdout-$i"`
	lmem=$(( `sed -n 's/.*, \([0-9+]\+\) bytes lmem,.*/\1/p' "$BUILD_LOGS/rnd_to_stdout-$i"` ))
	cmem=$(( `sed -n 's/.*, \([0-9+]\+\) bytes cmem.*/\1/p' "$BUILD_LOGS/rnd_to_stdout-$i"` ))
	smem=$(( `sed -n 's/.*, \([0-9+]\+\) bytes smem.*/\1/p' "$BUILD_LOGS/rnd_to_stdout-$i"` ))

	echo \"$rng_name\" $registers $lmem $cmem $smem
done
