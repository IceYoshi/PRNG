#!/bin/sh

header() {
	cat << !!
\\documentclass[12pt,a4paper,twoside]{article}

\\usepackage{color}
\\usepackage{array,listings,amsmath,amssymb,amsthm,graphicx}
\\bibliographystyle{alpha}

\\usepackage[utf8]{inputenc}

\\begin{document}
!!
}

footer() {
	cat << !!
\\end{document}
!!
}

filtered_test_results()
{
	grep -v '\<\(diehard_sums\|diehard_operm5\)\>' "$1"
}

table_header()
{
	num_lines=`filtered_test_results "$1" | wc -l | cut -d ' ' -f 1`
	echo -n '\\begin{tabular}{ '

	for i in `seq 0 $num_lines`; do
		echo -n " c |"
	done

	echo " }"

	echo 'RNG Name & '
	cut -d ',' -f 1 "$1" | (
		for i in `seq 2 $num_lines`; do
			read line
			printf '\\rotatebox{90}{%s} & ' "$line"
		done
		read line
		printf '\\rotatebox{90}{%s} \\\\ \n' "$line"
	) | sed 's/_/\\_/g'
	echo '\\hline'
}

print_result_cell()
{
	case "$1" in
	PASSED)
		echo -n '\\colorbox{green}{P}'
	;;
	WEAK)
		echo -n '\\colorbox{yellow}{W}'
	;;
	FAILED)
		echo -n '\\colorbox{red}{F}'
	;;
	*)
		echo "parse error"
	;;
	esac
}

table_body()
{
	num_lines=`filtered_test_results "$1" | wc -l | cut -d ' ' -f 1`
	rng_name=`echo "$1" | sed \
		-e 's/.*dieharder-\(.*\)/\1/' \
		-e 's/.*testu01small_crush-\(.*\)/\1/' \
		-e 's/.*testu01crush-\(.*\)/\1/' \
		-e 's/.*testu01big_crush-\(.*\)/\1/' \
		-e 's/_/\\\\_/g'`

	echo -n "$rng_name & "
	filtered_test_results "$1" \
	| sed -n "s/.*\(FAILED\|PASSED\|WEAK\).*/\1/p" \
	| (
		for i in `seq 2 $num_lines`; do
			read line
			print_result_cell "$line"
			echo -n '& '
		done
		read line
		print_result_cell "$line"
		echo ' \\\\ '
	)
	echo '\\hline'
}

table_footer()
{
	cat << !!
\\end{tabular}
!!
}

if [ $# -lt 1 ]; then
	echo "wrong number of arguments" >&2
	exit 1
fi

header
table_header "$1"

for i in $*; do
	echo $i; # slightly ugly...
done | sort | while read i; do
	table_body "$i"
done

table_footer
footer


