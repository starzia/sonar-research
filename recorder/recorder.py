#!/usr/bin/python
#-----------------------------------------
# Stephen Tarzia, starzia@northwestern.edu
#
# NOTE: the calibration proceduce depends on functions in extras.py
# so running this will not work.
#-----------------------------------------

# NOTE: numpy is an optional package in most linux distributions
import math, subprocess, time
from numpy import *
from numpy.fft import *
import sys,ossaudiodev,wave, audioop

#----------------------------------------------------------------------------
# Global variables whose values are modified by configuration file
#----------------------------------------------------------------------------
PHONE_HOME = 'decline' # default
LOG_START_TIME = 9999999999 # default
REC_DEV='/dev/dsp' # default

#----------------------------------------------------------------------------
# Constants
#----------------------------------------------------------------------------
SW_VER='1.0'
SPEAKER='right' # 'left'
TONE_LENGTH = 5 # ping length
IDLE_THRESH = TONE_LENGTH # period of input inactivity required, in seconds
CONFIDENCE_THRESH = 5 # cutoff for reliable presence detection
LOG_ADDR = 'steve@belmont.eecs.northwestern.edu'
SMTP_SERVER = 'belmont.eecs.northwestern.edu'
from os.path import expanduser
CONFIG_DIR_PATH = expanduser( '~/.sonarPM/' )
CONFIG_FILE_PATH = CONFIG_DIR_PATH + 'sonarPM.cfg'
LOG_FILE_PATH = CONFIG_DIR_PATH + 'log.txt'
TRIAL_PERIOD = 604800 # one week, in seconds
INT16_MAX = 32767
RATE=48000
#RATE=44100
TONE_VOLUME= 0.1 # one a scale from 0 to 1
REC_PADDING = 0.2 # amount of extra time to recordback to account for lack of
                # playback/record synchronization
BUFFER_LENGTH = 3 # three second record/playback buffer for aplay/arecord

SLEEP_TIME = 2
DEBUG = 0
TRAINING_TRIALS = 4
FFT_POINTS = 1024 # should be power of 2
WINDOW_SIZE=0.01 # energy spectrum window size, in seconds

set_printoptions(threshold=sys.maxint) # disable array summarization
#sampling interval
T_s = 1.0/RATE 
FFT_FREQUENCIES = FFT_POINTS/2

def open_rec_dev():
    """Open and return the audio recording device"""
    if DEBUG: print 'Opening audio device...'
    rec_dev = ossaudiodev.open( REC_DEV, 'r' )
    rec_dev.setparameters( ossaudiodev.AFMT_S16_LE, 1, RATE )

    # adjust mixer level for inputs to maximum volume
    inputs = [ ossaudiodev.SOUND_MIXER_RECLEV ]
    mixer=ossaudiodev.openmixer()
    for i in inputs:
        # if the recording device has such a channel, boost its level
        if mixer.controls() & (1 << i):
            mixer.set( i, (100,100) )
    mixer.close()
    
    return rec_dev

# sample format is S16_LE
def tone( duration=0.5, freq=440, delay=0, FADE_SAMPLES=44, 
          sample_rate=RATE ):
    """duration is in seconds and freq is the  tone pitch.  
    Delay, in seconds, is silent time added to before tone.
    FADE_SAMPLES is the length of the fade in and out periods.
    Returned as a sound buffer"""

    tone_length = int(math.floor( duration * sample_rate ))
    # the following will change for formats other than 16 bit signed
    # intialize array
    data = empty( (tone_length, 1), dtype=int16  )

    volumeScale = (TONE_VOLUME)*INT16_MAX
    for t in range( tone_length ):
        data[t] = int(round(
            volumeScale * math.sin( 2 * math.pi * t/sample_rate * freq ) ) )

    # fade in and fade out to prevent 'click' sound.
    for i in range( 0, FADE_SAMPLES ):
        attenuation = float(i)/FADE_SAMPLES
        data[i] = int( attenuation * data[i] )
        data[tone_length-1-i] = int( attenuation * data[tone_length-1-i] )

    data = prepend_silence( data, delay )
    return data.tostring() # tostring() serves as array2audio()

def prepend_silence( audio, silence_duration, sample_rate=RATE ):
    """prepend silence_duration seconds of silence to the audio buffer"""
    silence_length = int(math.floor( silence_duration * sample_rate ))
    data = empty( (silence_length + len(audio), 1), dtype=int16  )

    for t in range( silence_length ):
        data[t] = int(0)
    for t in range( len(audio) ):
        data[silence_length + t] = audio[t]
    return data 

def sleep_monitor():
    """powers off the computer's display"""
    if DEBUG: print "SLEEPING MONITOR"
    subprocess.Popen(["/usr/bin/xset", "dpms", "force", "standby"])

def idle_seconds():
    """returns the number of seconds since the X server has had HID input."""
    from os.path import abspath,dirname
    from os import getcwd
    script_dir = dirname(sys.argv[0])
    script_dir = abspath( script_dir + "/.." )
    script_file = script_dir + "/idle_detection/idle_detection"

    # this returns the stdout output of the program as a string
    idle_time = subprocess.Popen([script_file],stdout=subprocess.PIPE).communicate()[0]
    idle_time = int( idle_time )
    #print "  %d" % (idle_time,)
    return idle_time

def audio_length( audio_buffer ):
    """Returns the length, in seconds, of an audio buffer."""
    # 2* because each sample is 2 bytes
    return float(len( audio_buffer ))/(2*RATE) 
    
def audio_window( audio_buffer, length, start=0.0 ):
    """Returns a trimmed audio buffer, starting at 'start' seconds with given 
    'length'.  We use this for spectrum windows."""
    end = start + length
    # below, 2* is because each sample is 16 bits = 2 bytes
    return audio_buffer[ 2*int(start/T_s): 2*int(end/T_s) ]

def audio_repeat( audio_buffer, repetitions=2 ):
    new_buffer = ""
    for i in range( repetitions ):
        new_buffer += audio_buffer
    return new_buffer

def audio2array( audio_buffer ):
    """convert recording buffer into a numpy array"""
    return frombuffer( audio_buffer, int16 )

def my_fft( audio_buffer, N=FFT_POINTS ):
    """Fourier transform.  N is the number of points in the discretization.
    Returns frequency and amplitude arrays"""
    y = audio2array( audio_buffer )

    if DEBUG: print "Doing %d point FFT..." % (N,)
    Y=abs( fftshift(fft(y,N)) )

    # these are the frequencies for Y
    F=arange( -RATE/2, RATE/2, float(RATE)/len(Y) )

    # 'fold' negative frequencies onto positive
    i=1
    while F[i] < 0:
        Y[len(F)-i] = Y[len(F)-i] + Y[i]
        i += 1
    # remove negative frequencies
    F = F[i:]
    Y = Y[i:]
    return [F,Y]

def energy_spectrum( audio_buffer, N=FFT_POINTS ):
    """Gives the frequency energy spectrum of an audio buffer, without using
    windowing.  This is useful for periodic signals.  When using this
    function, it may be important to trim the buffer to precisely an integer
    multiple of the fundamental tone's period."""
    [F,Y] = my_fft( audio_buffer, N )
    # square amplitudes to get energies
    return [F,Y*Y]

def welch_energy_spectrum( audio_buffer, N=FFT_POINTS, window_size=WINDOW_SIZE,
                           overlap=0.5 ):
    """Estimates the frequency energy spectrum of an audio buffer by averaging
    the spectrum over a sliding rectangular window.  
    I think that this is an implementation of Welch's Method."""
    num_windows = int( floor( (audio_length( audio_buffer )-window_size)/ 
                              ((1-overlap)*window_size) ) )
    Y = zeros( (FFT_FREQUENCIES,) )
    for i in range( num_windows ):
        [F,Yi] = energy_spectrum( audio_window( audio_buffer, window_size, 
                                                i*(1-overlap)*window_size ) )
        Y += Yi
    return [F,Y]

def freq_index( freq ):
    """returns the spectrum index closest to a given frequency."""
    return round( (2.0*freq/RATE) * FFT_FREQUENCIES )

def freq_energy( audio_buffer, freq_of_interest, window_size=WINDOW_SIZE ):
    """returns the power/energy of the given time series data at the frequency
    of interest."""
    [F,Y] = welch_energy_spectrum( audio_buffer, FFT_POINTS, window_size )
    return Y[ freq_index(freq_of_interest) ]

def real_sleep( seconds ):
    """suspends execution for a given duration.  The standard time.sleep
    function may resume execution much sooner, if any signal is sent to the
    process.  This implementation is kind of inefficient and inaccurate"""
    end_time = time.time() + seconds
    while time.time() < end_time:
        time.sleep( 0.5 )

def write_audio( audio_buf, filename, sample_rate=RATE ):
    """parameter is a mono audio file but the output file is stereo with
    silent right channel"""
    # convert mono audio buffer to stereo
    # below, parameters are ( buffer, width, lfactor, rfactor)
    if SPEAKER=='right':
        audio_buf = audioop.tostereo( audio_buf, 2, 0, 1 )
    else: # SPEAKER=='left'
        audio_buf = audioop.tostereo( audio_buf, 2, 1, 0 )

    wfile = wave.open( filename, 'w' )
    wfile.setnchannels(2)
    wfile.setsampwidth(2) # two bytes == 16 bit
    wfile.setframerate(sample_rate)
    wfile.writeframes( audio_buf )
    wfile.close()

def read_audio( filename, stereo=True ):
    """reads a stereo audio file but returns a mono buffer"""
    wfile = wave.open( filename, 'r' )
    buf = wfile.readframes( wfile.getnframes() )
    # below, parameters are ( buffer, width, lfactor, rfactor)
    if stereo:
        return audioop.tomono( buf, 2, 1, 0 )
    else:
        return buf

def play_audio( audio_buffer ):
    """plays an audio clip.  This should return immediately after the 
    tone STARTS playback.  The returned process handle has a wait()
    function for blocking until the playback is completed."""
    if DEBUG: print 'Playing...'
    tmp_wav_file = CONFIG_DIR_PATH + "tone.wav"
    write_audio( audio_buffer, tmp_wav_file )

    # spawn background process to playback tone
    return subprocess.Popen(["aplay", "-q",
                             "--buffer-time=%d" % (BUFFER_LENGTH*1000000), 
                             tmp_wav_file])

def play_tone( tone_length, tone_freq ):
    """plays a sound tone of a given duration and pitch.  This should return
    immediately after the tone STARTS playback."""
    if DEBUG: print 'Generating sine tone...'
    blip = tone( tone_length, tone_freq, 0 )
    play_audio( blip )

def nonblocking_record_audio( seconds, filename ):
    """records audio for given duration using the arecord app.  
    returns a subprocess handle which has a wait() function."""
    # spawn process to record tone
    p = subprocess.Popen(["arecord", 
                          ("--device=%s" % "default"),
                          ("--rate=%d" % RATE), 
                          ("--duration=%f" % seconds), 
                          "--format=S16_LE",
                          "--channels=1",
                          "--quiet", 
                          "--buffer-time=%d" % (BUFFER_LENGTH*1000000),
                          filename ])
    return p

def record_audio( seconds ):
    """records audio for given duration."""
    if DEBUG: print 'Recording...'

    rec_dev = open_rec_dev()
    # below, 2* is for 2 bytes per sample (parameter is num of bytes to read)
    rec_bytes = 2*int( math.floor( RATE*seconds ) )
    rec = rec_dev.read( rec_bytes )

    # write recording to out.wav for debugging
    if DEBUG: write_audio( rec, 'out.wav' )

    # close audio devices
    # A freshly opened audio device seems to behave more predictably
    rec_dev.close()
    return rec

def recordback( audio ):
    """Plays audio while recording and return the recording"""
    play_audio( audio )
    # padding is to account for lack of record/playback synchronization
    p = nonblocking_record_audio( audio_length( audio ) + REC_PADDING, 
                                  "recordback.wav" )
    p.wait()
    return read_audio( "recordback.wav", False )

def downsample(vector, factor):
    """
    downsample(vector, factor):
        Downsample (by averaging) a vector by an integer factor.
    code from: http://mail.scipy.org/pipermail/numpy-discussion/2006-May/007961.html
    """
    if (len(vector) % factor):
        print "Length of 'vector' is not divisible by 'factor'=%d!" % factor
        return 0
    vector.shape = (len(vector)/factor, factor)
    return mean(vector, axis=1)


def measure_stats( audio_buffer, freq ):
    """Returns the mean and variance of the intensities of a given frequency
    in the audio buffer sampled in windows spread throughout the recording."""
    NUM_SAMPLES = 10 # the number of windows to create within the audio buffer
    DUTY = 0.2 # duty cycle for audio analysis
    polling_interval = audio_length( audio_buffer ) / NUM_SAMPLES
    
    intensities = []
    t=REC_PADDING
    while( t < audio_length( audio_buffer ) - polling_interval*DUTY ):
        intensities.append( freq_energy( audio_window( audio_buffer,
                                                       polling_interval*DUTY, 
                                                       t ),
                                         freq ) )
        t += polling_interval
    intensities = log10( array( intensities ) )
    variance = 1000*intensities.var()
    mean = 10*intensities.mean()
    return [ mean, variance ]

def log( message, logfile_name=LOG_FILE_PATH ):
    """adds timestamped message to the logfile"""
    t = time.time()
    str = "%s (%d): %s\n" % ( time.strftime("%Y/%m/%d %H:%M:%S",
                                            time.localtime(t)), 
                              int(t), message )
    logfile = open( logfile_name, 'a' ) # append to logfile
    logfile.write( str )
    logfile.close()

def term_handler( signum, frame ):
    """on power_management termination (by ctrl^c, kill, shutdown, etc),
    log this event."""
    log( "sonar power management terminated" )
    raise SystemExit

def power_management( freq, threshold ):
    """infinite loop that checks for idleness then shuts off monitor if
    sonar does not detect user"""
    # register signal handler for program termination
    from signal import signal, SIGTERM, SIGINT
    signal( SIGTERM, term_handler )
    signal( SIGINT, term_handler )

    global LOG_START_TIME
    LOG_START_TIME = log_start_time()

    log( "sonar power management began" )
    blip = tone( TONE_LENGTH, freq )
    while( 1 ):
        if( idle_seconds() > IDLE_THRESH ):
            log( "idle" )
            sleep=1
            for iter in ['first', 'second']:
                rec = recordback( blip )
                [ mean, variance ] = measure_stats( rec, freq )
                log( "%s sonar is %d" % (iter,variance) )
                print "var=%d\tmean=%d" % ( int(variance), int(mean) )
                if( variance > threshold or idle_seconds() < IDLE_THRESH ):
                    sleep=0
                    break
            if sleep:
                log( "standby" )
                sleep_monitor()
                real_sleep( 1 ) # wait for display state to be updated
                # wait until active again
                while( idle_seconds() < 0 ):
                    real_sleep( 2 )
                log( "active" )

        real_sleep( SLEEP_TIME )

        # if TRIAL_PERIOD has elapsed, phone home (if enabled)
        if ( time.time() - LOG_START_TIME > TRIAL_PERIOD ):
            phone_home()
            disable_phone_home() # prevent future log emails

def choose_ping_freq():
    """Prompt the user to find the best ping frequency.
    Generally, we want to choose a frequency that is both low enough to
    register on the (probably cheap) audio equipment but high enough to be
    inaudible."""
    # We start with 20khz and reduce it until we get a reading on the mic.
    print """
    This power management system uses sonar to detect whether you are sitting
    in front of the computer.  This means that the computer plays a very high
    frequency sound while recording the echo of that sound (and then doing
    some signal processing).  The calibration procedure that follows will try
    to choose a sound frequency that is low enough to register on your
    computer's microphone but high enough to be inaudible to you.  Note that
    younger persons and animals are generally more sensitive to high frequency
    noises.  Therefore, we advise you not to use this software in the company
    of children or pets."""
    silence = tone( TONE_LENGTH, 0 )
    silence_rec = recordback( silence )
    silence_reading=1
    blip_reading=1
    start_freq = 22000
    freq = start_freq
    # below, we subtract two readings because they are logarithms
    scaling_factor = 0.95
    print """
    Please press <enter> and listen carefully to continue with the 
    calibration."""
    sys.stdin.readline()
    while 1:
        freq *= scaling_factor
        blip = tone( TONE_LENGTH, freq )
        rec = recordback( blip )
        [ blip_reading, blip_var ] = measure_stats( rec, freq )
        [ silence_reading, silence_var ] = measure_stats( silence_rec, freq )
        if DEBUG: print "blip: %f silence: %f" % (blip_reading,silence_reading)
        print "Did you just hear a high frequency (%dHz) tone? [no]" %int(freq)
        line = sys.stdin.readline().strip()
        if line == 'yes' or line == 'Yes' or line == 'YES' or line == 'yea':
            freq /= scaling_factor
            break
    if freq >= start_freq:
        print "Your hearing is too good (or your speakers are too noisy)."
        print "CANNOT CONTINUE"
        log( "freq>=start_freq" )
        phone_home()
        raise SystemExit
    freq = int( freq )
    print "chose frequency of %d" % (freq,)
    return freq

def choose_ping_threshold( freq ):
    """Choose the variance threshold for presence detection by prompting
    the user."""
    from os import remove
    ping = tone( TONE_LENGTH, freq )
    calibration_file = CONFIG_DIR_PATH + 'calibration.dat'
    write_recordings( ping, calibration_file, TRAINING_TRIALS )
    rec = read_recordings( calibration_file )
    remove( calibration_file )
    mean = empty( (2,TRAINING_TRIALS) )
    var = empty( (2,TRAINING_TRIALS) )
    for p in [0,1]:
        for i in range(TRAINING_TRIALS):
            [ mean[p][i], var[p][i] ] = measure_stats( rec[p][i], freq )
    present_var = var[1].mean()
    not_present_var = var[0].mean()
    print "variances: present: %f not_present %f" % ( present_var, 
                                                      not_present_var )
    threshold = int( ceil( not_present_var ) )    
    confidence = present_var - not_present_var
    if confidence < CONFIDENCE_THRESH:
        print "Confidence is too low.  CANNOT CONTINUE!"
        log( "low confidence" )
        phone_home()
        raise SystemExit
    return [ threshold, confidence ]

def choose_recording_device():
    """Prompt the user to choose their preferred recording device.  This must
    be done before any audio recording can take place."""
    print "Audio playback will be done through the default ALSA device."
    print "Please enter your OSS recording device [/dev/dsp]:"
    line = sys.stdin.readline().rstrip()
    if line == "":
        line = '/dev/dsp'
    recording_device = line
    global REC_DEV
    REC_DEV = recording_device
    return recording_device

def warn_audio_level():
    """Plays a loud tone to persuade users to turn down their volume level"""
    beep = tone( 0.5, 1000 )
    print """
    A series of tones will now be played.  Please use these as references for
    adjusting your volume level.  The tones should be clearly audible but not
    uncomfortably loud.

    Press <enter> after you've adjusted the volume level.
    """
    def alrm_handler(signum, frame):
        return 

    from signal import signal, alarm, SIGALRM
    TIMEOUT = 2 # number of seconds your want for timeout
    while( 1 ):
        signal( SIGALRM, alrm_handler ) # catch ALRM with dummy handler
        alarm(TIMEOUT)
        try:
            play_audio( beep )
            real_sleep( 0.5 ) # must sleep because playback is non-blocking
            # wait for input
            sys.stdin.readline()
            # if user hits enter before timeout, then break out of loop
            break
        except:
            # timeout
            continue
            
def calibrate():
    """Runs some tests and then creates a configuration file in the user's
    home directory"""
    print """
==============================================================================
    In order to improve upon future versions of this software, we ask that you
    allow us to collect a log of software events.  No personal information, 
    including identifying information, will be collected.  The following is an
    example of the kind of imformation that would be collected:

2008/10/13 16:05:26 (1223931926): calibration frequency 17820 threshold 11 \
device /dev/dsp1
2008/10/13 16:08:41 (1223932121): sonar power management began
2008/10/13 16:08:48 (1223932128): idle
2008/10/13 16:08:55 (1223932135): first sonar is 6
2008/10/13 16:08:58 (1223932138): idle
2008/10/13 16:09:06 (1223932146): first sonar is 1
2008/10/13 16:09:08 (1223932148): idle
2008/10/13 16:09:16 (1223932156): first sonar is 6
2008/10/13 16:09:18 (1223932158): idle
2008/10/13 16:09:26 (1223932166): first sonar is 0
2008/10/13 16:09:26 (1223932166): standby
2008/10/13 16:09:26 (1223932166): active
2008/10/13 16:13:06 (1223932386): idle
2008/10/13 16:13:14 (1223932394): first sonar is 7
2008/10/13 16:14:01 (1223932441): idle
2008/10/13 16:14:08 (1223932448): first sonar is 6
2008/10/13 16:14:11 (1223932451): sonar power management terminated

    A log would be sent twice: first after calibration is completed and second
    after the first week of usage.  Sending the first message allows us to
    estimate the percentage of users that have "dropped out" before
    the one week period.  After you delete the local copy of your 
    logfile, there will be no way to associate an entry in our database with 
    your machine.  The source IP address of your messages will be scrubbed
    from our database.

    You may view the contents of the log at:
      %s""" % ( LOG_FILE_PATH, )
    phone_home = ""
    while phone_home != "send" and phone_home != "decline":
        print """
        Please type 'send' now to approve the automatic emailing of your log 
        file after calibration and one week from now.  
        Otherwise, type 'decline':"""
        phone_home = sys.stdin.readline().rstrip()
    global PHONE_HOME
    PHONE_HOME = phone_home

    print """
==============================================================================
    A short calibration procedure is now required in order to match
    the sonar system's parameters to your speakers, microphone, and the
    acoustics of your room.

    Please set the volume level on your speakers and on your soundcard mixer
    to a comfortable level.  You should NOT adjust these levels after 
    calibration.  You can, however, adjust the volume slider within your
    media player.

    If you switch rooms or want to change speaker volume levels you should 
    recalibrate the sonar.  This is done by deleting the file
    %s
    and restarting this application.

    Press <enter> to continue""" % (CONFIG_FILE_PATH,)
    sys.stdin.readline()
   
    # create configuration directory, if necessary
    from os.path import isdir
    from os import mkdir
    if not isdir( CONFIG_DIR_PATH ):
        mkdir( CONFIG_DIR_PATH )

    recording_device = choose_recording_device()
    warn_audio_level()
    freq = choose_ping_freq()
    [threshold,confidence] = choose_ping_threshold( freq )

    write_config_file( phone_home, recording_device, freq, threshold )
    log( "calibration: version %s frequency %d threshold %d device %s confidence %f" % 
         (SW_VER,freq,threshold,recording_device,confidence) )

def write_config_file( phone_home, recording_device, freq, threshold ):
    """Writes a configuration file with the passed values.  Note that the
    configuration directory must already exist."""
    # initialize a configuration object
    from ConfigParser import ConfigParser
    config = ConfigParser()
    config.add_section( 'general' )
    config.add_section( 'calibration' )

    # write configuration to a file
    config.set( 'general', 'phone_home', phone_home )
    config.set( 'general', 'recording_device', recording_device )
    config.set( 'calibration', 'frequency', freq )    
    config.set( 'calibration', 'threshold', threshold )
    config_file = open( CONFIG_FILE_PATH, 'w' )
    config.write( config_file )

def load_config_file():
    """Loads previous calibration data from config file or runs the
    calibration script if no data yet exists."""
    from os.path import exists
    if not exists( CONFIG_FILE_PATH ):
        print "Config file not found.  Calibration will follow."
        calibrate()
        # now config file should have been created
        return load_config_file()
    else:
        print "Config file found."
        global REC_DEV, PHONE_HOME
        from ConfigParser import ConfigParser
        config = ConfigParser()
        config.read( CONFIG_FILE_PATH )
        PHONE_HOME = config.get( 'general', 'phone_home' )
        REC_DEV = config.get( 'general', 'recording_device' )
        freq = int( config.get( 'calibration', 'frequency' ) )
        threshold = int( config.get( 'calibration', 'threshold' ) )
        return [freq, threshold]

def disable_phone_home():
    """rewrite configuration file to disable phoning home."""
    global PHONE_HOME,REC_DEV
    if PHONE_HOME == 'send':
        PHONE_HOME='decline'
        [freq,threshold] = load_config_file()
        write_config_file( PHONE_HOME, REC_DEV, freq, threshold )

def phone_home( dest_addr=LOG_ADDR, smtp_server=SMTP_SERVER ):
    """emails log.txt to us.  Also, disable future phoning home."""
    if PHONE_HOME != 'send':
        return # phone home disabled

    from smtplib import SMTP
    from_addr = 'user@localhost'
    msg = ("From: %s\r\nTo: %s\r\nSubject: sonarPM phone home\r\n\r\n"
           % (from_addr, dest_addr) )
    # append log file
    log_file = open( LOG_FILE_PATH, 'r' )
    for line in log_file:
        msg += line
    log_file.close()

    server = SMTP( smtp_server )
    server.set_debuglevel(0)
    server.sendmail(from_addr, dest_addr, msg)
    server.quit()
    print "Sent log email."

def log_start_time():
    """Parses the log file and returns the time of the first stamped message"""
    from os.path import exists
    # if log file has not yet been created return current time.
    if not exists( LOG_FILE_PATH ):
        return time.time()
    else:
        log_file = open( LOG_FILE_PATH, 'r' )
        line = log_file.readline()
        log_file.close()
        toks = line.split()
        return int( toks[2].strip('():') )

def main():
    [ freq, threshold ] = load_config_file()
    warn_audio_level()
    print "Sonar display power management has now begun. Hit <ctrl>-C to quit."
    power_management( freq, threshold )
    return

if __name__ == "__main__": main()
