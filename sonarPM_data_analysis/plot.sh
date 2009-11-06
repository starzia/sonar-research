#!/bin/bash
# This script does all the log file processing
#
# Note that using tmpfs (with many inodes for small files) speeds things up:
# sudo /bin/mount -t tmpfs -o size=2G,nr_inodes=200k,mode=0775,noatime,nodiratime tmpfs /mnt/tmpfs


# Parallelization functions
# wait until fewer jobs are running than number of processors
function queue {
  # number of processes for parallelization
  NP=15
  while [ `jobs -r|wc -l` -ge $NP ]; do
    sleep 1
  done
}
# wait until all jobs have finished
function barrier {
  while [ `jobs -r|wc -l` -ge 1 ]; do
    sleep 1
  done
}


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
  test $? == 1 && ( rm ${log}2 )
  # note, above, that we delete inconsistent logs
done
echo `ls users/*.log2 | wc -l` logs retained
# TODO: battery, AC, %laptop (used battery) vs desktop, correlate battery use w/session stats, displayTimeout settings 

# *.log2 files have been parsed by the awk script
# they exclude inconsistent logs, have the correct absolute timestamp,
# and they have statistics appended at the tail.


# copy head & tail of each log file for vital stats written by logger & our awk script.
# This is done to make repeated access to tail, below, more efficient
# Also, filter out logs less than one week long
# and those with little total_runtime
# and those with low ping gain
# *.log2tail files, are thus the stats from "good" users
rm -f users/*.log2tail
for log in users/*.log2; do
  head -n 10 $log > ${log}t
  tail -n 30 $log >> ${log}t
  total_duration="`cat ${log}t | grep -a -m 1 total_duration | cut -s -f3 -d\ `"
  total_runtime="`cat ${log}t | grep -a -m 1 total_runtime | cut -s -f3 -d\ `"
  ping_gain="`cat ${log}t | grep -a -m 1 ping_gain | cut -s -f3 -d\ `"
  if [ "$total_duration" ]; then
    if [ "$total_duration" -ge "604740" ]; then
      if [ "$total_runtime" ]; then
        if [ "$total_runtime" -ge "3600" ]; then
          if [ "$ping_gain" ]; then
            if [ "$(echo "$ping_gain >= 10.0"|bc)" -gt "0" ]; then
              # keep this log
              mv ${log}t ${log}tail
            fi
          fi
        fi
      fi
    fi
  fi
done
rm -f users/*.log2t
echo `ls users/*.log2tail | wc -l` good users


# plot log statistics CDFs, items joined with a + will be on same plot
echo "CDFs of log statistics"
for plot in \
 total_duration+total_runtime \
 false_sonar_cnt+false_timeout_cnt \
 sleep_sonar_cnt+sleep_timeout_cnt \
 sonar_cnt \
 sleep_sonar_len+sleep_timeout_len \
 sonar_timeout_ratio \
 active_len+passive_len \
 active_passive_ratio \
 sample_rate \
 ping_gain \
 displayTimeout \
; do
  echo "$PLT_COMMON set output '$plot.png'; set logscale x; set ylabel 'fraction of users with value <= x'; plot \\" > ${plot}.plt
  # for each line in the plot (separated by + chars)
  for stat in `echo $plot | sed "s/+/ /g"`; do
    > ${stat}.txt
    # get the data for that statistic from the end of all the log files.
    for log in users/*.log2tail; do
      guid="`echo $log | sed -e 's/\.log2tail//g' -e 's/users\///g'`"
      stat_value="`cat $log | grep -a -m 1 $stat | cut -s -f3 -d\ `"
      echo "$stat_value $guid" >> ${stat}.txt
    done
    cat ${stat}.txt | sort -n > ${stat}.txt2
    mv ${stat}.txt2 ${stat}.txt
    num_users=`cat ${stat}.txt | wc -l`
    echo "'${stat}.txt' using (\$1):(\$0/${num_users}) with lines, \\" >> ${plot}.plt
  done
  echo "(0) title '' with lines" >> ${plot}.plt
  gnuplot ${plot}.plt
done


# plot a histogram of model keywords
echo "Model keyword histogram"
for log in users/*.log2; do
  head $log | grep -a -m 1 model | sed 's/^.*model //g'
done | sort --ignore-case > models.txt
# remove DOS newlines
cat models.txt | sed "s/\r//g" >  models2.txt
cat models2.txt | sed "s/ /\n/g" | ./histogram.awk | sort -f | sed -e "s/[#:]//g" -e "s/\[//g" -e "s/\]//g" |sort -n -r -k 2 | head -n 50 > models.txt
rm models2.txt
echo "$PLT_COMMON set output 'models.png'; set logscale y; unset xtics; plot 'models.txt' using 1:2 with boxes;" |gnuplot


# plot CDF of freq response (show every fifth frequency)
echo "CDF of freq response"
for log in users/*.log2; do
  head $log | grep -a -m 1 response | sed 's/^.*response //g'
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


# TODO
# confusion matrix: total, avg, CDF

