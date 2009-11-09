set terminal postscript enhanced color
set output "4d.eps"
set logscale x
set logscale y
set logscale cb
set palette color negative
set view map
set title "all logs with greater than one hour of data"
set cblabel "runtime hours"
set xlabel "timeout sleep hours"
set ylabel "sonar sleep hours"

splot 'all_stats.txt' using ($11/3600):($10/3600):(($3/3600)):((1+(log($19))*100)**.5/10) with points pt 6 ps variable lt palette title "circle area is proportional to ping gain"
