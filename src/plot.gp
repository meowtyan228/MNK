reset
set terminal qt enhanced size 1200,800 font "Arial,14"

set xlabel "x"
set ylabel "y"
set grid
set key top left font "Arial,12"
set title "МНК"

set xrange [-5.5:5.5]
set yrange [-10:30]

plot \
    ARG1 using 1:2 with points pt 7 ps 0.5 lc rgb "black" title "Данные", \
    ARG2 using 1:2 with lines lw 2 lc rgb "red"   title "Истинный полином", \
    ARG2 using 1:3 with lines lw 2 lc rgb "blue"  title "Simple", \
    ARG2 using 1:4 with lines lw 2 lc rgb "green" title "QR", \
    ARG2 using 1:5 with lines lw 2 lc rgb "orange" dashtype 2 title "SVD"

pause -1