BEGIN{ 
  timestamp=0;
  install_time=0;
  app_start_time=0;
  total_runtime=0;
  new_file=0; /* set if the previous line had $2 == "file_end" */
  max_delta=15000000; /* ~173 days */
}
/./{
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

  if( $1 > max_delta ){
    /* absolute timestamp */

    if( install_time == 0 ){
      /* record initial time */
      install_time = $1;
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
  if(( $2 ~ /begin/ ) || ( $2 ~ /resume/ )){  app_start_time = timestamp; }
  if(( $2 ~ /end/ ) || ( $2 ~ /suspend/ )){
    total_runtime += timestamp - app_start_time;
    app_start_time = timestamp; /* just to be safe */ 
  }
}
END{
  printf( "%d total_duration %d\n", timestamp, timestamp-install_time );
  printf( "%d total_runtime %d\n", timestamp, total_runtime );
}
