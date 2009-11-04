BEGIN{ 
  /* state variables */
  timestamp=0;
  install_time=0;
  thread_start_time=0;
  new_file=0; /* set if the previous line had $2 == "file_end" */
  max_delta=15000000; /* ~173 days, this is just before Windows logs seem to start */
  last_sleep_sonar_time=0;
  last_sleep_timeout_time=0;
  /* user states: */
  sleeping_sonar=0;
  sleeping_timeout=0;
  passive=0;

  /* cumulative statistics */
  total_runtime=0;
  sleep_sonar_cnt=0;
  sleep_timeout_cnt=0;
  false_sonar_cnt=0;
  false_timeout_cnt=0;
  sleep_sonar_len=0;
  sleep_timeout_len=0;
  active_len=0;
  passive_len=0;
  sonar_cnt=0; /* number of sonar readings taken */
  ping_gain=0; /* from freq_response */
}
/./{
  /* ==================== SANITY CHECKS =======================*/
  /* test that first line is sane */
  if( NR == 1 ){
    if( $2 !~ /model/ ){
      exit( 1 ); /* exit with error code */
    }
  }
  /* test that first line in each log file is sane */
  if( new_file == 1 ){
    /* should have absolute, not relative, timestamp */
    if( $1 < max_delta ){
      exit( 1 );
    }
    new_file = 0;
  }
  if( $2 ~ /file_end/ ){
    new_file = 1;
  }

  /* ==================== FIX TIMESTAMP =======================*/
  if( $1 > max_delta ){
    /* absolute timestamp */
    if( install_time == 0 ){
      /* record initial time */
      install_time = $1;
      thread_start_time = $1; /* this shouldn't be necessary, but is */
    }
    if( $1 < timestamp ){
       /* we just went backward in time! */
       exit( 1 ); /* exit with error code */
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

  /* ==================== PARSE OPCODE =======================*/
  if(( $2 ~ /begin/ ) || ( $2 ~ /resume/ )){  thread_start_time = timestamp; }
  else if(( $2 ~ /end/ ) || ( $2 ~ /suspend/ )){
    total_runtime += timestamp - thread_start_time;
    thread_start_time = timestamp; /* just to be safe */ 
  }else if( $2 ~ /false/ ){
    if( $3 ~ /sleep/ ){
      false_sonar_cnt += 1;
    }else if( $3 ~ /timeout/ ){
      /* correct for repetitive sleep timeout bug */
      if( timestamp - last_sleep_timeout_time < 5 ){
        false_timeout_cnt += 1;
      }
    }
  }else if( $2 ~ /sleep/ ){
    if( $3 ~ /sonar/ ){
      sleep_sonar_cnt += 1;
      last_sleep_sonar_time = timestamp;
      sleeping_sonar = 1;
    }else if( $3 ~ /timeout/ ){
      /* must be the first message in the buggy repetitive sleep series */
      if( timestamp - last_sleep_timeout_time > 10 ){
        sleep_timeout_cnt += 1;
        last_sleep_timeout_time = timestamp;
        sleeping_timeout = 1;
      }
    }
  /* sonar reading */
  }else if(( $2 ~ /[0-9]/ ) && ( $3 ~ /[0-9]/ )){
    sonar_cnt += 1;
    passive = 1;
    /* if newly active-inactive, record sleep time */
    if( sleeping_sonar == 1 ){
      sleep_sonar_len += timestamp - last_sleep_sonar_time;
      sleeping_sonar = 0;
    }
    if( sleeping_timeout == 1 ){
      sleep_timeout_len += timestamp - last_sleep_timeout_time;
      sleeping_timeout = 0;
    }
  /* freq response */
  }else if( $3 ~ /response/ ){
    split( $22, fields, /[:,]/ );
    ping_gain = fields[2]/fields[3];
  }
}
END{
  printf( "%d total_duration %d\n", timestamp, timestamp-install_time );
  printf( "%d total_runtime %d\n", timestamp, total_runtime );
  printf( "%d false_sonar_cnt %d\n", timestamp, false_sonar_cnt );
  printf( "%d false_timeout_cnt %d\n", timestamp, false_timeout_cnt );
  printf( "%d sleep_sonar_cnt %d\n", timestamp, sleep_sonar_cnt );
  printf( "%d sleep_timeout_cnt %d\n", timestamp, sleep_timeout_cnt );
  printf( "%d sonar_cnt %d\n", timestamp, sonar_cnt );
  printf( "%d sleep_sonar_len %d\n", timestamp, sleep_sonar_len );
  printf( "%d sleep_timeout_len %d\n", timestamp, sleep_timeout_len );
  if( sleep_timeout_len > 0 ){
    printf( "%d sonar_timeout_ratio %d\n", timestamp, sleep_sonar_len/sleep_timeout_len );
  }else{
    printf( "%d sonar_timeout_ratio inf\n", timestamp );
  }
  printf( "%d active_len %d\n", timestamp, active_len );
  printf( "%d passive_len %d\n", timestamp, passive_len );
  if( passive_len > 0 ){
    printf( "%d active_passive_ratio %d\n", timestamp, active_len/passive_len );
  }else{
    printf( "%d active_passive_ratio inf\n", timestamp );
  }
  printf( "%d ping_gain %d\n", timestamp, ping_gain );
}
