#!/usr/bin/awk -f
#  from  O'reilly's bash cookbook

function max(arr, big){
  big = 0;
  for( i in user ){
    if( user[i] > big ){ big=user[i]; }
  }
  return big;
}

NF>0{
  /* note that we are not being case sensitive */
  user[tolower($1)]++
}

END{
  # for scaling
  maxm = max(user);
  for( i in user ){
    scaled = 50 * user[i] / maxm;
    printf "%17.17s [%8d]:", i, user[i];
    for( i=0; i<scaled; i++ ){
      printf "#";
    }
    printf "\n";
  }
}
