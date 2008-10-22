/* The following headers are provided by these packages on a Redhat system:
 * libX11-devel, libXext-devel, libScrnSaver-devel
 * There are also some additional dependencies for libX11-devel */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

#include <X11/extensions/scrnsaver.h>
#include <stdio.h>

int main(){
  Display *dis;
  XScreenSaverInfo *info;
  dis = XOpenDisplay((char *)0);
  Window win = DefaultRootWindow(dis);
  info = XScreenSaverAllocInfo();
  XScreenSaverQueryInfo( dis, win, info );

  /* if the screensaver is on return -1, else return idle time */
  if( info->state == ScreenSaverOn ){
    printf( "-1" );
  }else{
    printf( "%d", info->idle/1000 );
  }
}
