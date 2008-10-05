#!/usr/bin/awk -f
BEGIN{
  # occurence of last standby event
  last_standby=0
  last_standby_delay=0
  last_active=0
  last_idle=0

  # we use the following two values to determine the average idle time before
  # user is found to be not present.  Exclude false standby so these don't
  # skew the results.
  num_real_standbys=0
  delay_real_standbys=0

  num_false_standbys=0
}

/began/{
  gsub( "[():]","", $3 ) # extract number from third field
  last_idle = $3
}

/idle/{
  gsub( "[():]","", $3 ) # extract number from third field
  # if this idle message is not a repeat, ie it is the first one after the last
  # active burst
  if( last_active > last_idle ){
    last_idle = $3
  }
}

/active/{
  gsub( "[():]","", $3 ) # extract number from third field
  last_active=$3
  # test for false standby
  if( $3 - last_standby < 10 ){
    num_false_standbys++
    num_real_standbys--
    delay_real_standbys -= last_standby_delay
    printf( "(%d): standby %d marked false\n", $3, last_standby )
  }
}

/standby/{
  gsub( "[():]","", $3 ) # extract number from third field
  num_real_standbys++
  last_standby = $3
  last_standby_delay = $3 - last_idle
  delay_real_standbys += last_standby_delay

  printf( "(%d): standby delay is %d\n", $3, last_standby_delay )
}

END{
  printf( "num_real_standbys = %d\n", num_real_standbys )
  printf( "num_false_standbys = %d\n", num_false_standbys )
  avg_delay = delay_real_standbys/num_real_standbys
  printf( "avg standby idle time = %f\n", avg_delay )
}