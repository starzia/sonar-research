from recorder import *

#----------------------------------------------------------------------------
# Constants
#----------------------------------------------------------------------------
REQD_GAIN = 5 # this is the required minimum gain determining the ping freq
SWEEP_TONE_DURATION = 5
#THRESH = 1400000
THRESH = -1400000
INTERVALS_PER_TRAINING = 5
SLICE_FRAC=0.5 # the fraction of the total recording to use for central slice


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

def freq_energy2( audio_buffer, freq_of_interest ):
    """returns the power/energy of the given time series data at the frequency
    of interest."""
    [F,Y] = energy_spectrum( audio_buffer, FFT_POINTS )
    return Y[ freq_index(freq_of_interest) ]

def measure_stats2( audio_buffer, freq ):
    """Returns the mean and variance of the intensities of a given frequency
    in the audio buffer sampled in windows spread throughout the recording."""
    NUM_SAMPLES = 5 # the number of windows to create within the audio buffer
    DUTY = 1 # duty cycle for audio analysis
    polling_interval = audio_length( audio_buffer ) / NUM_SAMPLES
    
    intensities = []
    t=REC_PADDING
    while( t < audio_length( audio_buffer )-polling_interval ):
        intensities.append( freq_energy2( audio_window( audio_buffer,
                                                        polling_interval*DUTY, 
                                                        t ),
                                          freq ) )
        t += polling_interval
    intensities = log10( array( intensities ) )
    variance = 1000*intensities.var()
    mean = 10*intensities.mean()
    return [ mean, variance ]
