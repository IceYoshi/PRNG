#!/bin/sh

# this file generates makefile rules for different RNDs including targets to
# generate diagrams, tests, etc

if [ $# -ne 1 ]; then
	echo "wrong number of arguments" >&2
	exit 1
fi

# rnd name reaches until the first '.'
# Arguments are separated by dots
# '=' is problematic as make evaluates it. '-' will thus be expanded to '='. 

rnd_arguments=`echo "$1" | sed -e 's/^[^.]\+//' -e 's/-/=/g' -e 's/\./ -D/g' -e 's/\(-D[^=]\+=\)0\+/\1/g'`
rnd_name=`echo "$1" | sed 's/^\([^.]\+\).*/\1/'`

# debug 
#cat >&2 << !!
#rnd name: $rnd_name
#arguments to random number generator: $rnd_arguments
#
#!!

echo "# this file is auto generated"
echo

for i in ${STUB_DIR}/*.cu; do
	test_name=`basename $i .cu`

	cat << !!
build_targets += \$(RND_BIN)/${test_name}-${1}

\$(RND_BIN)/${test_name}-${1}: ${i} \$(RND_DIR)/${rnd_name}.hpp config.mk
	\$(NVCC) -o \$@ \$< -DRANDOM_NUMBER_GENERATOR=\"$rnd_name.hpp\" \
	-DRNG=$rnd_name \$(NVCCFLAGS) \$(WARNINGS) \$(LIBS) $rnd_arguments \
	--ptxas-options=-v > \$(BUILD_LOGS)/${test_name}-${1} 2>&1 \
	|| (cat \$(BUILD_LOGS)/${test_name}-${1}; false)
	@cat \$(BUILD_LOGS)/${test_name}-${1}
	@echo
	@echo

!!

done
