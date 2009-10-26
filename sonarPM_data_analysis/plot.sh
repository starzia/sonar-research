#!/bin/bash
# This script does all the log file processing

# download latest logs from belmont
sudo rsync -rv --delete root@belmont.cs.northwestern.edu:/home/sonar .
sudo chown -R `whoami`:`whoami` sonar
chmod -R +r sonar
# delete empty log files
for i in `find sonar -size 0`; do rm -f $i; done

find sonar/*.gz sonar/2009* > tmp_all_logs.txt
FIND_LOGS="cat tmp_all_logs.txt"
PLT_COMMON="set terminal png large size 1024,768; set grid;"


# plot a histogram of model keywords
for i in `$FIND_LOGS | grep "\.0\.log\.gz"`; do
  zcat -q $i | grep model | sed 's/^.*model //g'  # suppress zcat errors
done | sort --ignore-case > models.txt
# remove DOS newlines
cat models.txt | sed "s/\r//g" >  models2.txt
cat models2.txt | sed "s/ /\n/g" | ./histogram.awk | sort -f | sed -e "s/[#:]//g" -e "s/\[//g" -e "s/\]//g" |sort -n -r -k 2 | head -n 50 > models.txt
rm models2.txt
echo "$PLT_COMMON set output 'models.png'; set logscale y; unset xtics; plot 'models.txt' using 1:2 with boxes;" |gnuplot


# plot user retention (log indices)
for i in `seq 0 99`; do
  echo $i `$FIND_LOGS | grep "\.$i\.log\.gz"|wc -l`
done > log_indices.txt
echo "$PLT_COMMON set output 'log_indices.png'; set logscale y; \
set xlabel 'at least this many log files (one log per hour or app launch)'; \
set ylabel '# of users'; \
plot 'log_indices.txt' using 1:2 with lines;" |gnuplot


# plot CDF of freq response (show every fifth frequency)
for i in `$FIND_LOGS | grep "\.0\.log\.gz"`; do
  zcat -q $i | grep response | sed 's/^.*response //g' # suppress zcat errors
done > freq_responses.txt
total=`cat freq_responses.txt|wc -l`

echo "$PLT_COMMON set output 'freq_responses.png'; set logscale x; set key right outside; plot \\" > freq_responses.plt
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


# concatenate log files into one file for each user
mkdir users
for guid in `$FIND_LOGS | grep "\.0\.log\.gz" | sed -e 's/\.0\.log\.gz//g' -e 's/sonar.*\///g'`; do
  > users/${guid}.log
  for log in `$FIND_LOGS | grep $guid | sort -n --field-separator=. --key=2,2`; do
    zcat -q $log >> users/${guid}.log
  done
done


# replace timestamp time deltas w/ absolute time and 
# append some basic stats to end of log
for log in users/*.log; do
  cat $log | awk '
    BEGIN{ 
      timestamp=0;
      install_time=0;
      app_start_time=0;
      total_runtime=0;
      /* max_delta=3600*24*7; one week */
    }
    /./{
      if( $1 > ( timestamp - 60 ) ){ /* 60 second leeway for aynch disk access */
        /* absolute timestamp */
        if( timestamp == 0 ){
          /* record initial time */2
          install_time = $1;
        }
        timestamp = $1;
        print $0;	
      }else{
        /* relative timestamp, must rewrite line */
        timestamp = timestamp + $1;
        printf( "%d ", timestamp );
        for(i=2;i<=NF;i++){
          printf( "%s ", $i );
        }
        printf( "\t%s\n", $1 );
      }

      if(( $2 ~ /begin/ ) || ( $2 ~ /resume/ )){  app_start_time = timestamp; }
      if(( $2 ~ /end/ ) || ( $2 ~ /suspend/ )){
        total_runtime += timestamp - app_start_time;
        app_start_time = timestamp; /* just to be safe */ 
      }
      /* TODO: battery, AC */
    }
    END{
      printf( "%d total_duration %d\n", timestamp, timestamp-install_time );
      printf( "%d total_runtime %d\n", timestamp, total_runtime );
    }
  ' > ${log}2
done


# plot runtime and duration CDFs
for stat in total_duration total_runtime; do
  > ${stat}.txt
  for log in users/*.log2; do
    tail $log | grep $stat | cut -s -f3 -d\  >> ${stat}.txt
  done
  cat ${stat}.txt | sort -r -n > ${stat}.txt2
  mv ${stat}.txt2 ${stat}.txt
  echo "$PLT_COMMON set output '${stat}.png'; set logscale y; \
  set xlabel '${stat} at least this many hours'; \
  set ylabel '# of users'; \
  plot '${stat}.txt' using (\$1/3600):0 with lines;" |gnuplot
done

# filter out logs less than one week long

# TODO
# confusion matrix: total, avg, CDF

