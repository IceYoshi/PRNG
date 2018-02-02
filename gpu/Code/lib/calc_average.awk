#!/usr/bin/awk -f

/^[0-9]+/ {
	counter++;
	sum += $1;
}

END {
	printf "%f\n", sum / counter
}
