#!/usr/bin/python
#-----------------------------------------
# Stephen Tarzia, starzia@northwestern.edu
#-----------------------------------------
import recorder

#FREQ = 20112
FREQ = 19466

blip = recorder.read_audio( "/home/steve/power_management/recorder/19466.wav" )
variance = (recorder.log10(
        recorder.ping_loop_continuous_buf( blip, FREQ, 5, .05 ))).var()
print "%d" % (variance*100,)
