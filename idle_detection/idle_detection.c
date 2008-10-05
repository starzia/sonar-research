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
  printf( "%d\n", info->idle/1000 );
}
