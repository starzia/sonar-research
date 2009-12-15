#!/bin/bash
# This script does all the log file processing
#
# Note that using tmpfs (with many inodes for small files) speeds things up:
# sudo /bin/mount -t tmpfs -o size=2G,nr_inodes=200k,mode=0775,noatime,nodiratime tmpfs /mnt/tmpfs


# Parallelization functions
# wait until fewer jobs are running than number of processors
function queue {
  # number of processes for parallelization
  NP=8
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
# a single lock, implemented as a directory
function lock {
  while ! mkdir .lock 2> /dev/null; do
    sleep 0.01
  done
}
function unlock {
  rmdir .lock 2> /dev/null
}


if [ "$1" == "--download" ]; then
  # download latest logs from belmont
  echo "download latest logs"
  sudo rsync -r --delete root@belmont.cs.northwestern.edu:/home/sonar .
  sudo chown -R `whoami`:`whoami` sonar
  chmod -R +r sonar
  # delete empty log files
  for i in `find sonar -size 0`; do rm -f $i; done
fi

find sonar/*.gz sonar/2009* > tmp_all_logs.txt
FIND_LOGS="cat tmp_all_logs.txt"
PLT_COMMON="set terminal png large size 1280,1024; set grid; "

if [ "$1" == "--download" ]; then
  # concatenate log files into one file for each user
  echo "concatenate log files"
  mkdir users 2> /dev/null # store a single concatenated log for each user
  rm users/* 2> /dev/null
  mkdir logs 2> /dev/null # store each log fragment decompressed
  # concatenate logs in correct order
  for guid in `$FIND_LOGS | grep "\.0\.log\.gz" | sed -e 's/\.0\.log\.gz//g' -e 's/sonar.*\///g'`; do
    > users/${guid}.log
    for log in `$FIND_LOGS | grep $guid | sort -n --field-separator=. --key=2,2`; do
      filename=`echo $log | sed -e 's/\//-/g'`
      # check that we didn't decompress this log in a previous script run
      if [ ! -f logs/$filename ]; then
        gzip -t $log 2> /dev/null # verify that user's file upload succeeded
        test $? == 0 && (zcat -q $log > logs/$filename; dos2unix -q logs/$filename)
      fi
      if [ -f logs/$filename ]; then
        cat logs/$filename >> users/${guid}.log
      fi
      echo "0 file_end $log" >> users/${guid}.log
    done
  done
fi

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


# replace timestamp time deltas w/ absolute time and 
# append some basic stats to end of log
# also reject bad logs
rm users/*.log2 2> /dev/null
echo "AWK parse logs"
for log in users/*.log; do
  queue
  # spawn awk as a background process for parallelization
  awk -f parse_log.awk $log > ${log}2 &
  ## delete inconsistent logs
  #test $? == 1 && ( rm ${log}2 )
  # note, we take care of the above deletion later
done
barrier
echo `ls users/*.log2 | wc -l` logs retained
# TODO: battery, AC, %laptop (used battery) vs desktop, correlate battery use w/session stats

# *.log2 files have been parsed by the awk script
# they have the correct absolute timestamp,
# and they have statistics appended at the tail if the log is "good".


# copy head & tail of each log file for vital stats written by logger & our awk script.
# This is done to make repeated access to tail, below, more efficient
# Also, filter out logs less than one week long
# and those with little total_runtime
# and those with low ping gain
# *.log2tail files, are thus the stats from "good" users
echo "filter out bad logs"
rm -f users/*.log2tail
for log in users/*.log2; do
  head -n 10 $log > ${log}tail
  tail -n 30 $log >> ${log}tail
  total_runtime="`cat ${log}tail | grep -a -m 1 total_runtime | cut -s -f3 -d\ `"
  if [ ! $total_runtime ]; then
    rm ${log}tail
  fi
done
# delete blacklist entries
for guid in `cat blacklist.txt | cut -s -f1 -d\ `; do 
  rm -f users/${guid}.log2tail
done
echo `ls users/*.log2tail | wc -l` good users


# plot log statistics CDFs, items joined with a + will be on same plot
echo "CDFs of log statistics"
old_plot_list="\
 total_duration+total_runtime+log_lines \
 sonar_cnt+sleep_sonar_cnt+sleep_timeout_cnt+false_sonar_cnt+false_timeout_cnt \
 sleep_sonar_len+sleep_timeout_len+sleep_total_len \
 sonar_timeout_ratio \
 active_len+passive_len+absent_len \
 active_passive_ratio \
 present_absent_ratio \
 sample_rate \
 ping_gain \
 displayTimeout \
";
plot_list="\
 total_duration+total_runtime \
 sleep_sonar_cnt+sleep_timeout_cnt+false_sonar_cnt+false_timeout_cnt \
 sleep_sonar_len+sleep_timeout_len+sleep_total_len \
 active_passive_ratio \
 present_absent_ratio \
 ping_gain \
 displayTimeout \
 false_sonar_rate+false_timeout_rate \
 extra_sleep_rate \
 extra_sleep_fraction \
 extra_sleep_per_reading \
 SNR \
";
> all_stats.txt
rm *.stat.txt 2> /dev/null
# parse statistic values for all users
# create single file all_stats.txt with a column for each stat and 
# also individual files for each stat ${stat}.stat.txt
for log in guid users/*.log2tail; do
  guid="`echo $log | sed -e 's/\.log2tail//g' -e 's/users\///g'`"
  echo -n "${guid} " > all_stats_$guid.txt
  for plot in ${plot_list}; do
    for stat in `echo $plot | sed "s/+/ /g"`; do
      if [ "$guid" == "guid" ]; then
      # header row
        echo -n "${stat} " >> all_stats_$guid.txt
      else
        # get the data for that statistic from the end of all the log files.
        stat_value="`cat $log | grep -a -m 1 " ${stat}" | cut -s -f3 -d\ `"
        if [ "$stat_value" == ""  ]; then 
          stat_value="null"
        fi
        echo "${stat_value} ${guid}" >> ${stat}.stat.txt
        echo -n "${stat_value} " >> all_stats_$guid.txt
      fi
    done
  done
  echo >> all_stats_$guid.txt
  cat all_stats_$guid.txt >> all_stats.txt
  rm all_stats_$guid.txt
done

# set up all plt files and plot them
num_users=`cat all_stats.txt | wc -l`
for plot in ${plot_list}; do
  > ${plot}.plt
  echo "$PLT_COMMON set output '$plot.png'; set logscale x; set ylabel 'fraction of users with value <= x'; plot \\" > ${plot}.plt
  # for each line in the plot (separated by + chars)
  for stat in `echo $plot | sed "s/+/ /g"`; do
    cat ${stat}.stat.txt | sort -n > ${stat}.stat.txt2
    mv ${stat}.stat.txt2 ${stat}.stat.txt
    echo "'${stat}.stat.txt' using (\$1):((\$0+1)/${num_users}) with lines, \\" >> ${plot}.plt
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

# plot CDF of SNR
echo "CDF of SNR"
for log in users/*.log2; do
  tail -n 30 $log | grep -a -m 1 SNR | sed 's/^.*SNR //g'
done > SNR_all_users.txt
total=`cat SNR_all_users.txt|wc -l`

cat SNR_all_users.txt | sort -n > SNR_all_users.txt2
mv SNR_all_users.txt2 SNR_all_users.txt
echo "$PLT_COMMON set output 'SNR_all_users.png'; set logscale x; set xlabel 'SNR'; plot \\" > SNR_all_users.plt
  echo "'SNR_all_users.txt' using 1:((\$0+1)/$total) w l, \\" >> SNR_all_users.plt
echo "(0) title '' with lines" >> SNR_all_users.plt
gnuplot SNR_all_users.plt


# plot CDF of freq response (show every fifth frequency)
echo "CDF of freq response"
for log in users/*.log2; do
  head $log | grep -a -m 1 response | sed 's/^.*response //g'
done > freq_responses.txt
total=`cat freq_responses.txt|wc -l`

echo "$PLT_COMMON set output 'freq_responses.png'; set logscale x; set xrange [0.01:100000]; set key right outside; plot \\" > freq_responses.plt
for freq in `seq 0 5 35`; do
  freq_string=`echo "(15000*1.02^($freq))" | bc`
  # below, some records have a single value for the ratio, ignore these (NF<108)
  cat freq_responses.txt | sed "s/[:,]/ /g" | \
   awk "{noise=\$(3*$freq+2); \
         silence=\$(3*$freq+3); \
         if(silence==0 || NF < 108){print 0;}else{printf( \"%f\n\", noise/silence );}}" | \
   sort -n -k 1 > freq_${freq}.txt
  echo "'freq_${freq}.txt' using 1:((\$0+1)/$total) title '$freq_string Hz' with lines, \\" >> freq_responses.plt
done
echo "(0) title '' with lines" >> freq_responses.plt
gnuplot freq_responses.plt

# TODO
# confusion matrix: total, avg, CDF

