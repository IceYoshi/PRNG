#!/bin/sh

header() {
	cat << !!
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head> <title>report</title> </head>
<body>
!!
}

footer() {
	cat << !!
</body>
</html>
!!
}

filtered_test_results()
{
	grep -v '\<\(diehard_sums\|diehard_operm5\)\>' "$1"
}

table_header()
{
	cat << !!
<table border="1">
<tr>
!!
#	num_lines=`filtered_test_results "$1" | wc -l | cut -d ' ' -f 1`

	echo '<th>RNG Name</th>'
	filtered_test_results "$1" | cut -d ',' -f 1 | sed 's|\(.*\)|\t<th>\1</th>|'
	echo '</tr>'
}

print_result_cell()
{
	case "$1" in
	PASSED)
		printf '\t<td bgcolor="#00ff00">Passed</td>\n'
	;;
	WEAK)
		printf '\t<td bgcolor="#ffff00">Weak</td>\n'
	;;
	FAILED)
		printf '\t<td bgcolor="#ff0000">Failed</td>\n'
	;;
	*)
		echo "parse error"
	;;
	esac
}

table_body()
{
	rng_name=`echo "$1" | sed \
		-e 's/.*dieharder-\(.*\)/\1/' \
		-e 's/.*testu01small_crush-\(.*\)/\1/' \
		-e 's/.*testu01crush-\(.*\)/\1/' \
		-e 's/.*testu01big_crush-\(.*\)/\1/'`

	echo "<tr>"
	printf '\t<td>%s</td>\n' "$rng_name"

	filtered_test_results "$1" \
	| sed -n "s/.*\(FAILED\|PASSED\|WEAK\).*/\1/p" \
	| (
		while read line; do
			print_result_cell "$line"
		done
	)
	echo "</tr>"
}

table_footer()
{
	echo "</table>"
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


