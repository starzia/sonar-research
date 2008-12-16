#!/usr/bin/python
from extras import *
alsa_rec_devs = [['default:CARD=Intel',96000],
                 ['default:CARD=UA25',96000],
                 ['default:CARD=Headset',96000],
                 ['default:CARD=U0x46d0x8b0',96000]]
alsa_play_devs = [['default:CARD=Intel',96000],
                  ['default:CARD=SoundSticks',96000],
                  ['default:CARD=Headset',96000],
                  ['null',96000]]
try:
    local_user_study( 20000, 'trials.dat', alsa_rec_devs,alsa_play_devs, 60 )
except KeyboardInterrupt:
    print "cancelled by user"
    subprocess.Popen(["killall","arecord","aplay"])
except:
    subprocess.Popen(["killall","arecord","aplay"])
    print "Unexpected error:", sys.exc_info()[0]
    raise
