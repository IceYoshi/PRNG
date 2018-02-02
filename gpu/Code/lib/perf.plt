# set size 1.0, 1.0
set term postscript color solid
#set terminal svg
#size 7.2cm,4cm portrait enhanced color solid lw 1 

#"NimbusSanL-Regu,10" fontfile "/usr/share/texmf/fonts/type1/urw/helvetic/uhvr8a.pfb"

# default line styles (use with ls 1 .. ls 4)
set style line 1 lt rgb "#ea0707"
set style line 2 lt rgb "#0784ea"
set style line 3 lt rgb "#010591"
set style line 4 lt rgb "#cd07ea"

set style data histogram
set xtic rotate by -45 scale 0
set style fill solid border -1
# set auto x
set yrange [0:]
# set ytics 1
set ylabel 'seconds'
set output OUTPUT

plot DATA using ($2/1000000):xticlabels(1) notitle
