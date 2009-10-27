#!/bin/bash
# This script does all the log file processing

# download latest logs from belmont
echo "download latest logs"
sudo rsync -r --delete root@belmont.cs.northwestern.edu:/home/sonar .
sudo chown -R `whoami`:`whoami` sonar
chmod -R +r sonar
# delete empty log files
for i in `find sonar -size 0`; do rm -f $i; done

find sonar/*.gz sonar/2009* > tmp_all_logs.txt
FIND_LOGS="cat tmp_all_logs.txt"
PLT_COMMON="set terminal png large size 1024,768; set grid;"


# plot user retention (log indices)
echo "plot user retention"
for i in `seq 0 99`; do
  echo $i `$FIND_LOGS | grep "\.$i\.log\.gz"|wc -l`
done > log_indices.txt
echo "$PLT_COMMON set output 'log_indices.png'; \
set xrange [0.01:]; set logscale y; \
set xlabel 'at least this many log files (one log per hour or app launch)'; \
set ylabel '# of users'; \
plot 'log_indices.txt' using 1:2 with lines;" |gnuplot


# concatenate log files into one file for each user
echo "concatenate log files"
mkdir users 2> /dev/null
rm users/* 2> /dev/null
# concatenate logs in correct order
for guid in `$FIND_LOGS | grep "\.0\.log\.gz" | sed -e 's/\.0\.log\.gz//g' -e 's/sonar.*\///g'`; do
  > users/${guid}.log
  for log in `$FIND_LOGS | grep $guid | sort -n --field-separator=. --key=2,2`; do
    gzip -t $log 2> /dev/null # verify that user's file upload succeeded
    test $? == 0 && (zcat -q $log >> users/${guid}.log)
    echo "0 file_end $log" >> users/${guid}.log
  done
  dos2unix -q users/${guid}.log
done


# replace timestamp time deltas w/ absolute time and 
# append some basic stats to end of log
rm users/*.log2 2> /dev/null
echo "AWK parse logs"
for log in users/*.log; do
  awk -f parse_log.awk $log > ${log}2
  # delete inconsistent logs
  test $? == 1 && ( rm ${log}2 )
done
echo `ls users/*.log2 | wc -l` logs retained
# TODO: battery, AC, %laptop (used battery) vs desktop, correlate battery use w/session stats, displayTimeout settings 


# plot runtime and duration CDFs
echo "CDF of runtime and duration"
for stat in total_duration total_runtime; do
  > ${stat}.txt
  for log in users/*.log2; do
    guid="`echo $log | sed -e 's/\.log2//g' -e 's/users\///g'`"
    stat_value="`tail -n 30 $log | grep $stat | cut -s -f3 -d\ `"
    echo "$stat_value $guid" >> ${stat}.txt
  done
  cat ${stat}.txt | sort -r -n > ${stat}.txt2
  mv ${stat}.txt2 ${stat}.txt
done
 echo "$PLT_COMMON set output 'duration.png'; set logscale y; set logscale x; \
  set xlabel 'at least this many hours'; \
  set ylabel '# of users'; \
  plot 'total_duration.txt' using (\$1/3600):0 with lines, \
       'total_runtime.txt' using (\$1/3600):0 with lines;" |gnuplot


# plot a histogram of model keywords
echo "Model keyword historgram"
for log in users/*.log2; do
  head $log | grep model | sed 's/^.*model //g'
done | sort --ignore-case > models.txt
# remove DOS newlines
cat models.txt | sed "s/\r//g" >  models2.txt
cat models2.txt | sed "s/ /\n/g" | ./histogram.awk | sort -f | sed -e "s/[#:]//g" -e "s/\[//g" -e "s/\]//g" |sort -n -r -k 2 | head -n 50 > models.txt
rm models2.txt
echo "$PLT_COMMON set output 'models.png'; set logscale y; unset xtics; plot 'models.txt' using 1:2 with boxes;" |gnuplot


# plot CDF of freq response (show every fifth frequency)
echo "CDF of freq response"
for log in users/*.log2; do
  head $log | grep response | sed 's/^.*response //g'
done > freq_responses.txt
total=`cat freq_responses.txt|wc -l`

echo "$PLT_COMMON set output 'freq_responses.png'; set logscale x; set xrange [0.01:100000]; set key right outside; plot \\" > freq_responses.plt
for freq in `seq 0 5 35`; do
  freq_string=`echo "(15000*1.02^($freq))" | bc`
  cat freq_responses.txt | sed "s/[:,]/ /g" | \
   awk "{noise=\$(3*$freq+2); \
         silence=\$(3*$freq+3); \
         if(silence==0){print 0;}else{printf( \"%f\n\", noise/silence );}}" | \
   sort -n -k 1 > freq_${freq}.txt
  echo "'freq_${freq}.txt' using 1:(\$0/$total) title '$freq_string Hz' with lines, \\" >> freq_responses.plt
done
echo "(0) title '' with lines" >> freq_responses.plt
gnuplot freq_responses.plt


# filter out logs less than one week long

# TODO
# confusion matrix: total, avg, CDF

