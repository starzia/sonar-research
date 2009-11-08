BEGIN{
  /* constants */
  ACTIVE=0;
  PASSIVE=1;
  ABSENT=2;
  max_delta=15000000; /* ~173 days, this is just before Windows logs seem to start */
  /* state variables */
  timestamp=0;
  install_time=0;
  new_file=0; /* set if the previous line had $2 == "file_end" */
  /* start times for various states: */
  last_sleep_sonar_time=0;
  last_sleep_timeout_time=0;
  last_sleep_timeout_claim_time=0;
  thread_start_time=0;
  last_sonar_time=0;
  state_start_time=0;
  /* user states: */
  sleeping_sonar=0;
  sleeping_timeout=0;
  user_state=ACTIVE; /* 0=active, 1=passive, 2=absent */

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
  absent_len=0;
  sonar_cnt=0; /* number of sonar readings taken */
  ping_gain=0; /* from freq_response */
  displayTimeout=0;
}

function change_user_state( new_state, timestamp ){
  if( state_start_time == 0 ){ exit( 1 ) };
  if( user_state == ACTIVE ){
    active_len += timestamp - state_start_time;
    
  }else if( user_state == PASSIVE ){
    passive_len += timestamp - state_start_time;
  }else{
    absent_len += timestamp - state_start_time;
  }
  user_state = new_state;
  state_start_time = timestamp;

  /* if previously sleeping, state change means waking, so record sleep time */
  if( sleeping_sonar == 1 ){
    sleep_sonar_len += timestamp - last_sleep_sonar_time;
    sleeping_sonar = 0;
  }
  if( sleeping_timeout == 1 ){
    sleep_timeout_len += timestamp - last_sleep_timeout_time;
    sleeping_timeout = 0;
  }
}

function session_begin(){
  user_state = ACTIVE;
  state_start_time = timestamp;
  last_sonar_time = timestamp * 2; /* choose an arbitrary large value so that comparison condition below is not met until a real past sonar time is available */
  thread_start_time = timestamp;
  sleeping_sonar = 0;
  sleeping_timeout = 0;
}

function session_end(){
  /* simply ignore duplicate ends, as these are common */
  if( thread_start_time == 0 ){ return };
  change_user_state( ABSENT, timestamp ); /* record last state */
  total_runtime += timestamp - thread_start_time;
  /* resetting these allows us to find instances with missing "begin" */
  thread_start_time = 0;  
  state_start_time = 0;
  last_sonar_time = 0;
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
    if( $1 < timestamp ){
       /* we just went backward in time! */
       exit( 1 ); /* exit with error code */
    }
    if( timestamp != 0 && $1 > 2*timestamp ){
       /* we just went way forward in time! */
       exit( 1 ); /* exit with error code */
    }
    timestamp = $1;
    print $0;
    if( install_time == 0 ){
      /* record initial time */
      install_time = $1;
      /* this shouldn't be necessary, but is */
      session_begin();
    }	
  }else{
    /* relative timestamp, must rewrite line */
    timestamp = timestamp + $1;
    printf( "%d", timestamp );
    for(i=2;i<=NF;i++){
      printf( " %s", $i );
    }
    printf("\n");
  }

  /* ==================== PARSE OPCODE =======================*/
  /* sonar reading */
  if(( $2 ~ /[0-9]\.[0-9].*/ ) && ( $3 ~ /[0-9]\.[0-9].*/ )){
    sonar_cnt += 1;
    /* allow implicit "begin" */
    if( last_sonar_time == 0 ){
      session_begin();
    }
    /* if newly inactive */
    if( user_state == ACTIVE ){
      change_user_state( PASSIVE, timestamp );
    /* if newly active-inactive */
    }else if( timestamp - last_sonar_time > 3 ){
      /* two sessions are recorded */
      if( sleeping_sonar == 1 ){
        /* if sonar was shut off, then we have no real record of when user became active.  Just approximate by current time.  First active interval after sleep is therefore lost. */
        change_user_state( ACTIVE, timestamp );
      }else{
        /* if sonar was still running, then last active interval started when sonar stopped */
        change_user_state( ACTIVE, last_sonar_time );
      }
      change_user_state( PASSIVE, timestamp );
    }
    last_sonar_time = timestamp;
  /* active, note that there msgs are omitted in version 0.7 so we use hack above to estimate activity */
  }else if( $2 ~ /active/ ){
    change_user_state( ACTIVE, timestamp );
  }else if(( $2 ~ /begin/ ) || ( $2 ~ /resume/ )){
    session_begin();
  }else if(( $2 ~ /end/ ) || ( $2 ~ /suspend/ ) || ( $2 ~ /complete/ )){
    session_end();
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
      change_user_state( ABSENT, timestamp );
      sleep_sonar_cnt += 1;
      last_sleep_sonar_time = timestamp;
      sleeping_sonar = 1;
    }else if( $3 ~ /timeout/ ){
      /* must be the first message in the buggy repetitive sleep series */
      if( timestamp - last_sleep_timeout_claim_time > 10 ){
        change_user_state( ABSENT, timestamp );
        sleep_timeout_cnt += 1;
        last_sleep_timeout_time = timestamp;
        sleeping_timeout = 1;
      }
      last_sleep_timeout_claim_time = timestamp;
    }
  }else if( $2 ~ /displayTimeout/ ){
    displayTimeout = $3;
  /* freq response */
  }else if( $2 ~ /response/ ){
    /* we don't have the freq reponse at 22khz, so we avg the two straddling values */
    split( $22, fields, /[:,]/ );
    split( $23, fields2, /[:,]/ );
    if( fields[3] != 0 && fields2[3] != 0){
      ping_gain = ( (fields[2]/fields[3]) + (fields2[2]/fields2[3]) )/2;
    }
  }
}
END{
  session_end();

  /* reject unfit logs */
  if( total_runtime < 3600 ){
    exit( 1 );
  }

  /* print log stats */
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
    printf( "%d sonar_timeout_ratio %f\n", timestamp, sleep_sonar_len/sleep_timeout_len );
  }else{
    printf( "%d sonar_timeout_ratio inf\n", timestamp );
  }
  printf( "%d active_len %d\n", timestamp, active_len );
  printf( "%d passive_len %d\n", timestamp, passive_len );
  printf( "%d absent_len %d\n", timestamp, absent_len );
  if( passive_len > 0 ){
    printf( "%d active_passive_ratio %f\n", timestamp, active_len/passive_len );
  }else{
    printf( "%d active_passive_ratio inf\n", timestamp );
  }
  if( absent_len > 0 ){
    printf( "%d present_absent_ratio %f\n", timestamp, (active_len+passive_len)/absent_len );
  }else{
    printf( "%d present_absent_ratio inf\n", timestamp );
  }
  printf( "%d ping_gain %f\n", timestamp, ping_gain );
  printf( "%d displayTimeout %d\n", timestamp, displayTimeout );
}
