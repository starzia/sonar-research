#!/usr/bin/python
#-----------------------------------------
# Stephen Tarzia, starzia@northwestern.edu
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
REQD_GAIN = 5 # this is the required minimum gain determining the ping freq
TONE_LENGTH = 5 # ping length
IDLE_THRESH = TONE_LENGTH # period of input inactivity required, in seconds
CONFIDENCE_THRESH = 10 # cutoff for reliable presence detection
LOG_ADDR = 'starzia@northwestern.edu'
SMTP_SERVER = 'hecky.it.northwestern.edu'
from os.path import expanduser
CONFIG_DIR_PATH = expanduser( '~/.sonarPM/' )
CONFIG_FILE_PATH = CONFIG_DIR_PATH + 'sonarPM.cfg'
LOG_FILE_PATH = CONFIG_DIR_PATH + 'log.txt'
TRIAL_PERIOD = 604800 # one week, in seconds
INT16_MAX = 32767
RATE=44100
TONE_VOLUME= 0.1 # one a scale from 0 to 1
REC_PADDING = 0.2 # amount of extra time to recordback to account for lack of
                # playback/record synchronization
SWEEP_TONE_DURATION = 5
#THRESH = 1400000
THRESH = -1400000
SLEEP_TIME = 2
DEBUG = 0
TRAINING_TRIALS = 4
INTERVALS_PER_TRAINING = 5
FFT_POINTS = 1024 # should be power of 2
WINDOW_SIZE=0.01 # energy spectrum window size, in seconds
SLICE_FRAC=0.5 # the fraction of the total recording to use for central slice

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
    inputs = [ ossaudiodev.SOUND_MIXER_MIC,
               ossaudiodev.SOUND_MIXER_RECLEV,
               ossaudiodev.SOUND_MIXER_DIGITAL1,
               ossaudiodev.SOUND_MIXER_IGAIN,
               ossaudiodev.SOUND_MIXER_LINE,
               ossaudiodev.SOUND_MIXER_LINE1,
               ossaudiodev.SOUND_MIXER_PHONEIN ]
    mixer=ossaudiodev.openmixer()
    for i in inputs:
        # if the recording device has such a channel, boost its level
        if mixer.controls() & (1 << i):
            mixer.set( i, (100,100) )
    mixer.close()
    
    return rec_dev

def log_spacing_interval( start_freq, end_freq, num_steps ):
    """returns the factor that needs to be applied to start_freq to reach
    end_freq after num_steps.  Useful for dividing a frequency range
    into logarithmically spaced steps"""
    interval = math.pow( float(end_freq)/start_freq, 1.0/num_steps )
    return interval

def log_range( lowest_freq, highest_freq, num_steps ):
    """returns an exponentially spaced range of numbers.  Useful for creating
    aurally-spaced tones."""
    frequencies = []
    f = lowest_freq
    # factor to increase frequency in each step
    interval = log_spacing_interval( lowest_freq, highest_freq, num_steps )
    while f < highest_freq:
        frequencies.append( f )
        # below we have a dynamic frequency scaling factor
        f *= interval
    return frequencies

# sample format is S16_LE
def tone( duration=0.5, freq=440, delay=0 ):
    """duration is in seconds and freq is the
    tone pitch.  Returned as a sound buffer"""
    tone_length = int(math.floor( duration * RATE ))
    # the following will change for formats other than 16 bit signed
    # intialize array
    data = empty( (tone_length, 1), dtype=int16  )

    volumeScale = (TONE_VOLUME)*INT16_MAX
    for t in range( tone_length ):
        data[t] = int(round(
            volumeScale * math.sin( 2 * math.pi * t/RATE * freq ) ) )

    data = prepend_silence( data, delay )
    return data.tostring()

def log_sweep_tone( duration, start_freq, end_freq, delay=0 ):
    """duration is in seconds and freqs are the
    tone pitch.  Returned as a sound buffer.
    TODO: this fcn output contains weird beats"""

    tone_length = int(math.floor( duration * RATE ))
    # the following will change for formats other than 16 bit signed
    # intialize array
    data = empty( (tone_length, 1), dtype=int16  )

    # use a loud volume for sweep
    volumeScale = (0.8)*INT16_MAX

    interval = log_spacing_interval( start_freq, end_freq, tone_length )
    freq = start_freq
    for t in range( tone_length ):
        freq *= interval
        #freq = start_freq * math.pow( interval, t ) 
        data[t] = int(round(
            volumeScale * math.sin( 2 * math.pi * float(t)/RATE * freq ) ) )

    data = prepend_silence( data, delay )
    return data.tostring()

def lin_sweep_tone( duration, start_freq, end_freq, delay=0 ):
    """duration is in seconds and freqs are the
    tone pitch.  Returned as a sound buffer"""

    tone_length = int(math.floor( duration * RATE ))
    # the following will change for formats other than 16 bit signed
    # intialize array
    data = empty( (tone_length, 1), dtype=int16  )

    # use a loud volume for sweep
    volumeScale = (0.8)*INT16_MAX

    b = ( start_freq - end_freq ) / (2*duration)
    for t in range( tone_length ):
        time = float(t)/RATE
        data[t] = int(round( volumeScale * 
                     math.cos( 2*math.pi*( start_freq*time - b*time*time ) ) ))

    data = prepend_silence( data, delay )
    return data.tostring()

def prepend_silence( audio, silence_duration ):
    """prepend silence_duration seconds of silence to the audio buffer"""
    silence_length = int(math.floor( silence_duration * RATE ))
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

def highpass( audio_buf, RC ):
    """Return RC high-pass filter output samples, given input samples,
    and time constant RC.
    The method is time-domain simulation of an RC filter."""
    x = audio2array( audio_buf )
    y = empty( (x.size,), dtype=int16 )
    alpha = float(RC) / (RC + T_s)
    y[0] = x[0]
    for i in range( 1, x.size ):
        y[i] = alpha*(y[i-1] + x[i] - x[i-1])
    return y

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

def energy_versus_time( audio_buffer, freq, window_size=0.01, N=FFT_POINTS ):
    """returns the energy of the given frequency in the buffer as a function
    of time [T,e].  window_size is the granularity of time sampling, in
    seconds"""
    total_length = audio_length( audio_buffer )
    T = arange( 0, RATE*(total_length-window_size), RATE*window_size )/RATE
    E = []
    for t in T:
        clip = audio_window( audio_buffer, window_size, t )
        E.append( freq_energy( clip, freq, window_size/2 ) )
    return [T,array(E)]

def freq_index( freq ):
    """returns the spectrum index closest to a given frequency."""
    return round( (2.0*freq/RATE) * FFT_FREQUENCIES )

def freq_energy( audio_buffer, freq_of_interest, window_size=WINDOW_SIZE ):
    """returns the power/energy of the given time series data at the frequency
    of interest."""
    [F,Y] = welch_energy_spectrum( audio_buffer, FFT_POINTS, window_size )
    return Y[ freq_index(freq_of_interest) ]

def freq_energy_goertzel( audio_buffer, freq_of_interest, 
                          window_size=WINDOW_SIZE ):
    """returns the power/energy of the given time series data at the frequency
    of interest.  Uses Goertzel algorithm. Code adapted from:
    http://en.wikipedia.org/wiki/Goertzel_algorithm
    It turns out that this is less efficient than doing the entire FFT.
    FFT is O(N log N) where N is the number of *buckets*, a small number."""
     # normalized_frequency is measured in cycles per sample
    normalized_frequency = float(freq_of_interest) / RATE
    x = audio2array( audio_buffer )

    s_prev = 0
    s_prev2 = 0
    coeff = 2*math.cos( 2*math.pi * normalized_frequency )
    for x_n in x:
      s = x_n + coeff*s_prev - s_prev2
      s_prev2 = s_prev
      s_prev = s    
    return s_prev2*s_prev2 + s_prev*s_prev - coeff*s_prev2*s_prev

def ascii_plot( vec, width=79 ):
    """prints an ascii bargraph.  Works only for positive numbers and there
    must not be any NaNs.  This fcn is obsoleted by matplotlib/pylab."""
    max = float( vec.max() )
    print "Normalized to maximum of %f" %(max,)
    tick_spacing = ( 10.0 ** floor( log10( max ) ) ) *width/max 
    minor_tick_spacing = tick_spacing / 10
    for i in vec:
        #print each bar, normalized to maximum length
        next_tick = tick_spacing
        next_minor_tick = minor_tick_spacing
        for j in range( int( float(width) * i / max ) ):
            if (j+1) > next_tick:
                sys.stdout.write("|")
                next_tick += tick_spacing
                next_minor_tick += minor_tick_spacing
            else:
                if( minor_tick_spacing > 3 ): # don't draw minor ticks if close
                    if (j+1) > next_minor_tick:
                        sys.stdout.write(":")
                        next_minor_tick += minor_tick_spacing
                    else:
                        sys.stdout.write("#")
                else:
                    sys.stdout.write("#")
        print ""

def unequal_dot( vec1, vec2 ):
    """Dot product of vectors with unequal length.  Just trim the longer one 
    first.  Only works for int data type (regular dot only works for floats)"""
    # we right shift the values of each vector first to avoid overflow in the 
    # multiplication.
    return ( (vec1[:len(vec2)]>>8) * (vec2[:len(vec1)]>>8) ).sum()

def cross_corellation( vector, source_vector, offset=100, N=100 ):
    """returns a vector of cross_correlation for increaing lags.
    TODO: start with negative lag rather than zero."""
    C = zeros( (N,) )

    for i in range( N ):
        C[i] = unequal_dot( source_vector, vector )
        # negative correlation terms are irrelevant
        if C[i] < 0: C[i] = 1
        
        # add padding to the source
        source_vector = hstack(  ( zeros((offset,), dtype=int16), 
                                   source_vector ) )
    return C

def cross_corellation_au( audio_buf, source_audio_buf, offset=100, N=100 ):
    return cross_corellation( audio2array(audio_buf), 
                              audio2array(source_audio_buf),
                              offset, N )

def autocorellation( vector, offset=100, N=100 ):
    """cross-corellation of a vector with itself"""
    return cross_corellation( vector, vector, offset, N )

def autocorellation_au( audio_buf, offset=100, N=100 ):
    return autocorellation( audio2array(audio_buf), offset, N )

def trim_front_silence( audio_buf ):
    """Trims the silent beginning of the audio buffer.  We consider silence to
    be sound that is many times softer than the loudest buffer point."""
    SILENCE_FACTOR = 10
    # include some padding before the peak we detected
    SILENCE_PADDING = 10 # number of samples
    arr = audio2array( audio_buf )
    max = arr.max()
    i=0
    while i < arr.size and arr[i] < max/SILENCE_FACTOR:
        i += 1
    if( i<SILENCE_PADDING ): i=SILENCE_PADDING
    return arr[i-SILENCE_PADDING:].tostring()

def real_sleep( seconds ):
    """suspends execution for a given duration.  The standard time.sleep
    function may resume execution much sooner, if any signal is sent to the
    process.  This implementation is kind of inefficient and inaccurate"""
    end_time = time.time() + seconds
    while time.time() < end_time:
        time.sleep( 0.5 )

def write_audio( audio_buf, filename ):
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
    wfile.setframerate(RATE)
    wfile.writeframes( audio_buf )
    wfile.close()

def read_audio( filename ):
    """reads a stereo audio file but returns a mono buffer"""
    wfile = wave.open( filename, 'r' )
    buf = wfile.readframes( wfile.getnframes() )
    # below, parameters are ( buffer, width, lfactor, rfactor)
    return audioop.tomono( buf, 2, 1, 0 )

def play_audio( audio_buffer ):
    """plays an audio clip.  This should return
    immediately after the tone STARTS playback."""
    if DEBUG: print 'Playing...'
    tmp_wav_file = CONFIG_DIR_PATH + "tone.wav"
    write_audio( audio_buffer, tmp_wav_file )

    ## spawn background process to playback tone
    subprocess.Popen(["/usr/bin/aplay", "-q", tmp_wav_file])
    #play_dev.write( audio_buffer )

def play_tone( tone_length, tone_freq ):
    """plays a sound tone of a given duration and pitch.  This should return
    immediately after the tone STARTS playback."""
    if DEBUG: print 'Generating sine tone...'
    blip = tone( tone_length, tone_freq, 0 )
    play_audio( blip )

def record_audio( seconds ):
    """records audio for given duration.  returns an audio buffer."""
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
    return record_audio( audio_length( audio ) + REC_PADDING )

def measure_ping( freq ):
    """Plays a tone while recording and then measures the energy of that freq
    in the recording.  Return the energy value"""
    blip = tone( TONE_LENGTH, freq )
    return measure_buf( blip, freq )

def slice_buf( buf, freq, slice_frac=SLICE_FRAC ):
    """Slices audio from the center. slice_frac is a fraction of the entire 
    length.  Using a slice is intended to eliminate silence in the beginning 
    and end of the buffer."""
    slice_len = slice_frac * audio_length( buf )

    # Trim the recording to the slice.
    # first adjust the slice to be an integer multiple of the fundamental freq
    period = 1. / freq
    slice_len = period * floor( slice_len / period )
    # cut out a central slice of the audio
    slice_start = ( audio_length( buf ) - slice_len ) / 2.
    buf = audio_window( buf, slice_len, slice_start )
    return buf
    
def measure_buf( buf, freq ):
    """same as measure_ping, but takes audio buffer argument.
    Slices off front and back portion of buffer before processing."""
    rec = recordback( buf )
    rec = slice_buf( rec, freq )

    if DEBUG: print 'Processing...'
    # measure energy of recording
    energy = freq_energy( rec, freq )
    return energy

def trim_to_range( frequencies,Y, start_freq,stop_freq ):
    """trims frequencies and Y to the given range.
    Y, as always is Y[present_bool][trials][frequencies]"""
    i=0
    while i<len(frequencies) and frequencies[i]<start_freq:
        i += 1
    j=i
    while j<len(frequencies) and frequencies[j]<=stop_freq:
        j += 1
    return [ frequencies[i:j], Y[:,:,i:j] ]
    

def best_freq( frequencies, Y ):
    """Returns the frequency with highest average gain minus standard 
    deviation.  Y is a 3D array of measured amplitudes:
    Y[present_bool][trials][frequencies]"""
    # We consider frequencies with high degrees of variance between trials to
    # be noisy.
    m = Y.mean( axis=1 ) # mean along the "trials" axis
    s_dev = Y.std( axis=1 ) # standard deviation along the "trials" axis
    s_dev = s_dev.mean( axis=0 ) # combine std_dev for both present and not
    s_dev = s_dev/m[0] # normalize std_dev to mean
    # Choose the frequency with the highest gain - normalized_std_dev
    gain = (m[1]/m[0])
    adjusted_gain = gain - s_dev
    best_f = frequencies[ adjusted_gain.argmax() ]
    best_g = gain[ adjusted_gain.argmax() ]
    print column_stack( (frequencies,gain,s_dev,adjusted_gain) )
    print "Highest noise-adjusted avg gain of %f at %dHz" % (best_g,best_f)
    return best_f

def training_discrete( lowest_freq=30, highest_freq=17000 ):
    """finds the frequency with highest degree of amplification or attentuation
    by guiding the user through several training steps.
    The parameters define the frequency search boundaries"""
    frequencies =log_range( lowest_freq,highest_freq,INTERVALS_PER_TRAINING )
    # make recordings
    # Y is the FFT data for all trials both present and not-present
    Y = zeros( (2,TRAINING_TRIALS,len(frequencies)) )

    for present in  [ 1, 0 ]:
        if present:
            print "Press <enter> and sit still until further notice"
            sys.stdin.readline()
        else:
            print "Press <enter> and walk away until further notice"
            sys.stdin.readline()
            real_sleep( 3 )

        for trial in range( TRAINING_TRIALS ):
            for f in range( len(frequencies) ):
                # notice that we are filling in only one cell of the 3D array
                # with each recording.
                Y[ present, trial, f ] = measure_ping( frequencies[f] )
    # find best frequencies
    amp_freq = best_freq( frequencies, Y )

    # recur if the resolution of these trials was too coarse
    interval = log_spacing_interval( lowest_freq, highest_freq,\
                                     INTERVALS_PER_TRAINING )
    if interval < 1.02:
        return amp_freq
    else:
        return training_discrete( amp_freq/interval, amp_freq*interval )

def training( lowest_freq=30, highest_freq=17000 ):
    """finds the frequency with highest degree of amplification or attentuation
    The parameters define the frequency search boundaries"""
    tone = lin_sweep_tone( SWEEP_TONE_DURATION, lowest_freq, highest_freq )

    # Y is the FFT data for all trials both present and not-present
    Y = empty( (2,TRAINING_TRIALS,FFT_FREQUENCIES) )
    for present in  [ 1, 0 ]:
        if present:
            print "Press <enter> and sit still until further notice"
            sys.stdin.readline()
        else:
            print "Press <enter> and walk away until further notice"
            sys.stdin.readline()
            real_sleep( 3 )
        for trial in range( TRAINING_TRIALS ):
            # notice that we are filling in an entire column of frequency
            # responses with a single recording.
            rec = recordback( tone )
            rec = slice_buf( rec, highest_freq ) # passed freq is meaningless
            [ F, y ] = welch_energy_spectrum( rec )
            Y[present,trial] = y
            real_sleep( 1 ) # short break btwn trials ensure audio dev avail
    # disregard frequencies outside of the played range when choosing best
    [F,Y] = trim_to_range( F,Y, lowest_freq, highest_freq )
    return best_freq( F, Y )

def ping_loop( freq, length=TONE_LENGTH ):
    # the pinging loop
    prev_energy = 9999999
    blip = tone( length, freq )
    while 1:
        energy = measure_buf( blip, freq )
        print "Recording energy at %dHz was %f dB" % (freq, log10(energy))
        columns = int( log10( energy ) * 5 )
        for i in range( columns ):
            if (i+1) % 5:
                sys.stdout.write("-")
            else:
                sys.stdout.write("+")
        print ""
        if energy < THRESH and prev_energy < THRESH:
            sleep_monitor()
            prev_energy = energy

        # sleep until next measurement
        if DEBUG: print 'Sleeping...'
        real_sleep( SLEEP_TIME )
    return

def measure_stats( audio_buffer, freq ):
    """Returns the mean and variance of the intensities of a given frequency
    in the audio buffer sampled in windows spread throughout the recording."""
    NUM_SAMPLES = 10 # the number of windows to create within the audio buffer
    DUTY = 0.2 # duty cycle for audio analysis
    polling_interval = audio_length( audio_buffer ) / NUM_SAMPLES
    
    intensities = []
    t=REC_PADDING
    while( t < audio_length( audio_buffer ) ):
        intensities.append( freq_energy( audio_window( audio_buffer,
                                                       polling_interval*DUTY, 
                                                       t ),
                                         freq ) )
        t += polling_interval
    intensities = log10( array( intensities ) )
    variance = 1000*intensities.var()
    mean = 10*intensities.mean()
    return [ mean, variance ]

def ping_loop_continuous( freq, length=60, polling_interval=0.4 ):
    """Plays a tone $length seconds long while periodically measuring the
    energy of that frequency recorded at the microphone.
    Polling_interval is the time period after which the energy is calculated.
    returns an array of the energies at each sampling period."""
    # round to an integer because we will be repeating a one second sample
    freq = int( freq ) 
    blip = tone( 1, freq )
    blip = audio_repeat( blip, int( length ) )
    return ping_loop_continuous_buf( blip, freq, length, polling_interval )

def ping_loop_continuous_buf( blip, freq, length, polling_interval ):
    """blip is an audio buffer containing the ping."""
    start_time = time.time()
    play_audio( blip )
    E = []
    while( time.time() - start_time < length ):
        rec = record_audio( polling_interval )
        energy = freq_energy( rec, freq )
        E.append( energy )
        time.sleep( 0.2 ) # sleep to prevent 100% CPU utilization
        #print "%f" % (energy,)
    return array(E)

def test():
    freq = 1119.726562
    trials = 5
    blip = tone( 1, freq )
    Y = zeros( (trials,FFT_FREQUENCIES) )
    for i in range( trials ):
        rec = recordback( blip )
        rec = slice_buf( rec, freq )
        [F,Y[i]] = energy_spectrum( rec )
        real_sleep( SLEEP_TIME )
    F = F[:len(F)/10] # ignore the top 90% of freqs in printout
    Y = Y[:,:len(F)]
    print array2string( column_stack( (F,Y[0:trials].T)), precision=2 )

def recording_xcorr( rec, ping, ping_period, offset=1 ):
    """measures the cross correlation of the recording and ping, after cutting
    out the head and tail of the recording"""
    length = audio_length( ping )
    period_samples = int( RATE*ping_period / offset )

    # confine to a window
    rec = audio_window( rec, length/2.0, REC_PADDING+0.1 )

    cross_corr = cross_corellation_au( rec, ping, offset, 2*period_samples )

    # trim the cross corellation data to a window starting at its peak
    # value and lasting for just over one period length
    peak_loc = cross_corr.argmax()
    # if the largest peak happens to be at the beginning of the last
    #  (incomplete) period, then choose the previous corresponding peak
    if peak_loc+period_samples >= cross_corr.size:
        peak_loc = peak_loc-period_samples
    cross_corr = cross_corr[peak_loc:]
    cross_corr = cross_corr[:period_samples]
    return cross_corr        

def measure_ping_CTFM( ping, length=1, offset=1 ):
    """creates a transmission of given length from a sequence of given ping
    audio buffers.  Then plays and records this CTFM tone and returns the
    cross corellation of the recording relative to the CTFM tone.  The
    returned cross corellation is aligned to the highest peak and cut off after
    one period of the tone."""
    period = audio_length( ping )
    # repeat ping to extend to length duration
    ping = audio_repeat( ping, int(ceil(length/period))  )

    buf = recordback( ping )
    return recording_xcorr( buf, ping, period, offset )

def test_CTFM( ping_length = 1, ping_period = 0.01, freq_start = 20000, 
               freq_end = 2000 ):
    """plots cross corellations (relative to ping) for two sets of recordings,
    present and not present"""
    ping = lin_sweep_tone( ping_period, freq_start, freq_end )

    # prime the recorder so it will start immediately when needed
    buf = recordback( ping )

    cc_samples = int(floor((ping_period/T_s)))
    cc = empty( (2,TRAINING_TRIALS,cc_samples) )
    for present in  [ 1, 0 ]:
        if present:
            print "Press <enter> and sit still until further notice"
            sys.stdin.readline()
        else:
            print "Press <enter> and walk away until further notice"
            sys.stdin.readline()
            real_sleep( 3 )
        for trial in range( TRAINING_TRIALS ):
            
            cc[present][trial] = measure_ping_CTFM( ping, ping_length ) 
            real_sleep( 1 )
    return cc

def plot_cc( cc ):
    """for use with the return value of test_CTFM()"""
    from pylab import subplot,plot
    subplot( 211 )
    plot( cc[0].T )
    subplot( 212 )
    plot( cc[1].T )

def CTFM_scope( ping_length = 1, ping_period = 0.01, freq_start = 20000,
                freq_end = 2000 ):
    """gives an interactive view of the cross correlations"""
    from pylab import plot,ion,draw,ylim
    OFFSET = 1 # reduces crosscorellation resolution to speed up display

    ping = lin_sweep_tone( ping_period, freq_start, freq_end )
    cc = measure_ping_CTFM( ping, ping_length, OFFSET )
    ion() # turn on pylab interactive mode for dynamic updates
    limit = cc.max()+zeros( (10,) )
    (line,) = plot( cc )
    (limit_line,) = plot( limit )
    max_y = 0
    while 1:
        cc = measure_ping_CTFM( ping, ping_length , OFFSET )
        if( cc.max() > max_y ): 
            max_y = cc.max()
            ylim( 0, max_y ) # adjust y axis extent
        line.set_ydata( cc )
        limit_line.set_ydata( cc.max()+zeros( (10,) ) )
        draw()
        real_sleep( 0.2 )

def plot_audio( audio_buf ):
    from pylab import plot
    plot( audio2array( audio_buf ) )

def write_recordings( ping, filename, num_trials=TRAINING_TRIALS ):
    """Make echo recordings and store to a file for later use"""
    from pickle import dump
    recordings = [[],[]]
    for present in  [ 1, 0 ]:
        if present:
            print "Press <enter> and sit in front of the computer until further notice"
            sys.stdin.readline()
        else:
            print "Press <enter>, leave the computer, and stay away until you hear a beep"
            sys.stdin.readline()
            real_sleep( 3 )
        for trial in range( num_trials ):
            recordings[present].append( recordback( ping ) )
            real_sleep( 1 )
    # beep to recall user to computer
    play_audio( tone( 0.5, 2000 ) )
    file = open( filename, 'wb' )
    dump( recordings, file )
    file.close()

def read_recordings( filename ):
    """Read back an array of recordings (for both present and not present)"""
    from pickle import load
    file = open( filename, 'rb' )
    recordings = load( file )
    file.close()
    return recordings

def make_cc( recordings, ping, ping_period ):
    """makes an array of cross correlations from an array of recordings"""
    cc_samples = int(floor((ping_period/T_s)))
    trials = len(recordings[0])
    cc = empty( (2,trials,cc_samples) )
    for i in [0,1]:
        for j in range( trials ):
            cc[i][j] = recording_xcorr( recordings[i][j], ping, ping_period )
    return cc

def sweep_freq( trials=TRAINING_TRIALS, 
               length=10, start_freq=100, end_freq=20000 ):
    """measures the energy spectrum of a frequency sweep recording"""
    tone = lin_sweep_tone( length, start_freq, end_freq )
    Y = zeros( (2,trials,FFT_FREQUENCIES) )
    for present in  [ 1, 0 ]:
        if present:
            print "Press <enter> and sit still until further notice"
            sys.stdin.readline()
        else:
            print "Press <enter> and walk away until further notice"
            sys.stdin.readline()
            real_sleep( 3 )
        for i in range( trials ):
            rec = recordback( tone )
            [F,Y[present,i]] = welch_energy_spectrum( rec )
    [F,Y] = trim_to_range( F, Y, start_freq, end_freq )
    return [F,Y]

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
            rec = recordback( blip )
            [ mean, variance ] = measure_stats( rec, freq )
            log( "first sonar is %d" % (variance,) )
            print "var=%d\tmean=%d" % ( int(variance), int(mean) )
            if( variance <= threshold and idle_seconds() > IDLE_THRESH ):
                rec = recordback( blip )
                [ mean, variance ] = measure_stats( rec, freq )
                log( "second sonar is %d" % (variance,) )
                print "var=%d\tmean=%d" % ( int(variance), int(mean) )
                if( variance <= threshold and idle_seconds() > IDLE_THRESH ):
                    log( "standby" )
                    sleep_monitor()
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
        print "Did you just hear an annoying high frequency tone? [no]"
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

    print """
    A two second tone will now be played.  Please use this as a reference for
    adjusting your volume level.  If the tone is very loud, then please turn
    down the volume level!
    """
    play_tone( 2, 1000 )

    # create configuration directory, if necessary
    from os.path import isdir
    from os import mkdir
    if not isdir( CONFIG_DIR_PATH ):
        mkdir( CONFIG_DIR_PATH )

    recording_device = choose_recording_device()
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
    [phone_home,rec_dev,freq,threshold] = load_config_file()
    write_config_file( 'decline', rec_dev, freq, threshold )

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
    print "Sonar display power management has now begun. Hit <ctrl>-C to quit."
    power_management( freq, threshold )
    return

if __name__ == "__main__": main()
