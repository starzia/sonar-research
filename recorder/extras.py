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
    """Return RC high-pass filtered audio buffer, given input buffer,
    and time constant RC.
    The method is time-domain simulation of an RC filter."""
    x = audio2array( audio_buf )
    y = empty( (x.size,), dtype=int16 )
    alpha = float(RC) / (RC + T_s)
    y[0] = x[0]
    for i in range( 1, x.size ):
        y[i] = alpha*(y[i-1] + x[i] - x[i-1])
    return y.tostring()

def lowpass( audio_buf, RC ):
    """Return RC low-pass filtered audio buffer, given input buffer,
    and time constant RC.
    The method is time-domain simulation of an RC filter."""
    x = audio2array( audio_buf )
    y = empty( (x.size,), dtype=int16 )
    alpha = float(RC) / (RC + T_s)
    y[0] = x[0]
    for i in range( 1, x.size ):
        y[i] = y[i-1] + alpha * (x[i] - y[i-1])
    return y.tostring()

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
    # TODO: is this neccessary?
    rec = audio_window( rec, length/2.0, REC_PADDING+0.1 )

    cross_corr = cross_corellation_au( rec, ping, offset, 2*period_samples )

    # trim the cross corellation data to a window starting at its peak
    # value and lasting for just over one period length
    peak_loc = cross_corr.argmax()
    # if the largest peak happens to be at the beginning of the last
    #  (incomplete) period, then choose the previous corresponding period
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


def CTFM_gnuplot( ping_length = 1, ping_period = 0.01, freq_start = 20000,
                  freq_end = 2000, OFFSET=1 ):
    """gives an interactive view of the cross correlations,
    OFFSET can be used to reduce xcorr resolution to speed up display"""
    # set up the plots
    gnuplot = subprocess.Popen(["gnuplot"], shell=True, \
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    print >>gnuplot.stdin, "set terminal x11;"

    ping = lin_sweep_tone( ping_period, freq_start, freq_end )
    # autocorrelation
    ac = cross_corellation_au( ping, ping, OFFSET, int( ping_period*RATE ) )
    max_y = 0
    recording=0
    j=0
    while 1:
        j+=1
        # start non-blocking playback
        full_ping = audio_repeat( ping, int(ceil(ping_length/ping_period))  )
        play_audio( full_ping )
    
        # start non-blocking record
        arecord = subprocess.Popen(["arecord", 
                                    ("--device=%s"%"default"),
                                    ("--rate=%d"%RATE), 
                                    ("--duration=%f"%ping_length), 
                                    "--format=S16_LE",
                                    "--channels=1",
                                    "--quiet", 
                                    "out.wav"])
        
        # process previous iteration's recording, if not 1st iteration
        if( j > 1 ):
            recording = read_audio( "out.wav", False ) #its mono 
            # calculate cross correlation
            cc = recording_xcorr( recording, full_ping, ping_period, OFFSET )
            if( cc.max() > max_y ): 
                max_y = cc.max()
            # calculate mixed frequency spectrum
            mixed = audio2array( recording ) * audio2array( full_ping )
            ## low-pass filter the mixed signal so we get only the
            ## difference signal, not the sum as well.
            #filtered = lowpass( mixed.tostring(), 1.0/(freq_start) )
            #[ F, M ] = welch_energy_spectrum( filtered )
            [ F, M ] = welch_energy_spectrum( mixed )
            [ F, Y ] = welch_energy_spectrum( recording )
        else:
            cc=ac
            [F,Y,M]=[array([0,1]),array([0,1]),array([0,1])]

        # plot it
        print >>gnuplot.stdin, "set multiplot layout 2,2 title 'CTFM sonar %dHz to %dHz in %f sec (%dHz sample rate)';" % (freq_start, freq_end, ping_period, RATE )
        print >>gnuplot.stdin, "set xrange [*:*];" # autoscale range

        print >>gnuplot.stdin, "set title 'ping autocorrelation';"
        print >>gnuplot.stdin, "set xlabel 'lag (samples)'";
        print >>gnuplot.stdin, "plot '-' using ($1/%f) with lines \
          title 'autocorrelation';" % ac.max()
        for i in ac:
            print >>gnuplot.stdin, i
        print >>gnuplot.stdin, "EOF"
        print >>gnuplot.stdin, "set title 'recording-ping cross correlation';"
        print >>gnuplot.stdin, "plot '-' using ($1/%f) with lines \
          title 'recording';" % cc.max()
        for i in cc:
            print >>gnuplot.stdin, i
        print >>gnuplot.stdin, "EOF"
        
        print >>gnuplot.stdin, "set title 'recording spectrum';"
        print >>gnuplot.stdin, "set xlabel 'frequency (Hz)"
        #print >>gnuplot.stdin, "set logscale x;"
        print >>gnuplot.stdin, "plot '-' using 1:($2/%f) with lines \
          title 'spectrum';" % Y.max()
        for i in range( size(Y) ):
            print >>gnuplot.stdin, F[i], Y[i]
        print >>gnuplot.stdin, "EOF"
        #print >>gnuplot.stdin, "unset logscale x;"

        print >>gnuplot.stdin, "set title 'mixed frequency spectrum';"
        # below, limit view to possible freq differences only
        print >>gnuplot.stdin, "set xrange [0:%f];" %abs(freq_start - freq_end)
        print >>gnuplot.stdin, "plot '-' using 1:($2/%f) with lines \
          title 'spectrum';" % M.max()
        for i in range( size(M) ):
            print >>gnuplot.stdin, F[i], M[i]
        print >>gnuplot.stdin, "EOF"
        print >>gnuplot.stdin, "unset multiplot;"

        # wait for recording to finish
        arecord.wait()



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

def bartlett( audio_buffer, freq, NUM_SAMPLES=10 ):
    """ The first returned is Bartlett's method result w/ NUM_SAMPLES windows.
    It estimates the energy of a given frequency in the audio buffer.
    The second returned value is what we actually used in the USENIX09 paper,
    which is the variance among those windows.
    """
    DUTY = 1 # duty cycle for audio analysis
    polling_interval = audio_length( audio_buffer ) / NUM_SAMPLES
    
    intensities = []
    t=0

    while( t < audio_length( audio_buffer )-polling_interval ):
        intensities.append( freq_energy2( audio_window( audio_buffer,
                                                        polling_interval*DUTY, 
                                                        t ),
                                          freq ) )
        t += polling_interval
    intensities = log10( array( intensities ) )
    return [ intensities.mean(), intensities.var() ]

def bartlett_recording( audio_buffer, args=[20000,500,10] ):
    """Divides the buffer into the specified number of divisions.
    Each division is further divided into NUM_SAMPLES pieces when Bartlett's
    method is called.
    The frequency's energy is calculated in each piece and then the variance
    and mean is calculated within each division.

    @return array of num_divisions intensity values"""
    freq=args[0]
    num_divisions=args[1]
    NUM_SAMPLES=args[2]

    intensities = empty( (num_divisions,) )

    window_size = audio_length( audio_buffer ) / num_divisions
    # move non-overlapping sliding window accross recording
    for i in range(num_divisions):
        window_buf = audio_window( audio_buffer, window_size,
                                   i*window_size )
        intensities[i] = bartlett( window_buf, freq, NUM_SAMPLES )[0]
    return intensities


def process_recording( audio_buffer, freq, num_divisions, NUM_SAMPLES=10 ):
    """Divides the buffer into the specified number of divisions.
    Each division is further divided into NUM_SAMPLES pieces when Bartlett's
    method is called.
    The frequency's energy is calculated in each piece and then the variance
    and mean is calculated within each division.
    We return the average of the means and variances across all divisions and
    also the minimum and maximum variances among all divisions.
    The second return value is what was reported in the USENIX09 submittion
    results, however, what we described in the Analysis section was actuall
    the last return value."""
    stats = empty( (num_divisions, 2) )

    window_size = audio_length( audio_buffer ) / num_divisions
    for i in range(num_divisions):
        window_buf = audio_window( audio_buffer, window_size,
                                   i*window_size )
        stats[i] = bartlett( window_buf, freq, NUM_SAMPLES )
    # return [mean of mean, mean of var, max-var, min-var, var of mean]
    return [ stats[:,0].mean(), stats[:,1].mean(), 
             stats[:,1].max(), stats[:,1].min(),
             stats[:,0].var() ]


def randomly_order( my_list ):
    """returns a random permutation of the passed list."""
    from random import random
    ordering = []
    for i in range(len(my_list)):
        random_index = int( floor( random() * len(my_list) ) )
        ordering.append( my_list.pop( random_index ) )
    return ordering

def local_user_study( ping_freq=20000, data_filename="trials.dat",
                      alsa_rec_dev_list=[["default",44100]],
                      alsa_play_dev_list=[["default",44100]],
                      seconds_per_echo=60 ):
    """This is the script for the local user study.  Randomly orders the study
    tasks, and helps the administrator to guide the user through them while
    playing and recording the ping.
    Alsa devices are listed under /proc/asound/devices.  use format 'hw:0,0'
    and you can pass a list of several devices to record from simultaneously.
    With each device you also pass the desired sampling rate."""
    from time import strftime

    start_time_str = strftime('%x %X')
    # generate user id number
    id=-1
    f=open(data_filename, 'r')
    for line in f:
        id = line.split()[0]
    f.close()
    id = int(id) + 1
    print "user id is %d" % id

    tasks = [ "word processor","watch video","telephone survey","word search",
              "absent" ]
    # choose a random ordering of the first four tasks
    task_order = randomly_order( range( len(tasks)-1 ) )
    task_order.append( 4 ) # absent is always second to last.
    print "task_order is going to be %s" % task_order

    # choose a random ordering of the playback devices
    play_dev_order = randomly_order( range(len(alsa_play_dev_list)) )
    
    # generate ping files at different possible sample rates
    print "generating ping wav files..."
    sample_rates = [44100,48000,96000]
    for rate in sample_rates:
        ping = tone( 1, ping_freq, 0,44,rate )
        ping = audio_repeat( ping, seconds_per_echo +10 )
        write_audio( ping, ("ping_%d.wav"%rate), rate )

    # user tasks
    for i in task_order:
        print "Next task is %s.  Press <enter> to begin." % tasks[i]
        sys.stdin.readline()
        print " stalling before recording starts..."
        # wait a little while to let the user settle down
        real_sleep( 5 )

        # start recording on all devices
        recorder_list = [] # this the the list of recording subrocesses
        for j in range( len(alsa_rec_dev_list) ):
            rec_filename = "%03d.%s.%d.wav" % (id,i,j)
            recorder_list.append( subprocess.Popen(["arecord", 
                                          ("--device=%s"%alsa_rec_dev_list[j][0]),
                                          ("--rate=%d"%alsa_rec_dev_list[j][1]), 
                                          "--format=S16_LE",
                                          "--channels=1",
                                          "--quiet", 
                                          rec_filename]) )
        print "Microphones activated."

        # cycle through the playback devices
        for j in play_dev_order:
            # spawn background process to playback ping tone
            player = subprocess.Popen(["aplay",
                                       ("--device=%s" % alsa_play_dev_list[j][0]),
                                       "--quiet",
                                       ("ping_%d.wav" %alsa_play_dev_list[j][1])])
            print "Playback device %s activated." % alsa_play_dev_list[j][0]
            time.sleep( seconds_per_echo )
            # stop the player
            subprocess.Popen([ "kill", ("%s"%player.pid) ])

        # stop each recorder
        for recorder in recorder_list:
            subprocess.Popen(["kill", ("%s"%recorder.pid)])
        print "Recording stopped."

    # write new data for this user
    f=open(data_filename, 'a')
    f.write( "%03d %s %s %s\n" % (id,start_time_str,task_order,play_dev_order) )
    f.close()


# constants
NUM_STATES=5
NUM_PLAY_DEVS=4
NUM_REC_DEVS=4

def process_all_recordings( data_directory, 
                            processing_func, processing_func_args ):
    """reads the recordings produced by local_user_study(), 

    data_directory is a string indicating where trials.dat and the wav files
    are stored.
    processing_func is the function with which to process the audio
    buffers.

    returns a 4d list of processing_func results:
        recording[user_id][state][rec_dev][play_dev]"""

    # parse trials.dat
    import re
    trials_filename = "%s/trials.dat" % data_directory
    f = open( trials_filename, 'r' )
    users = []
    p_dev_mapping = []
    for line in f:
        m = re.search( "(.{3}).*\[.*\] \[(\d), (\d), (\d), (\d)\]", line )
        users.append( m.group(1) )
        p_dev_mapping.append( list( m.group(2,3,4,5) ) )
    num_users = len( users )

    # intialize what will become a 4D recording list
    recording = [] 

    # read wav files
    for user_i in range( num_users ):
        recording.append([]) # add a new user column
        for state in range( NUM_STATES ):
            recording[user_i].append([]) # add a new state column 
            for r_dev in range( NUM_REC_DEVS ):
                recording[user_i][state].append([]) # add a new r_dev column 
                filename= "%s/%s.%d.%d.wav"%(data_directory,users[user_i],
                                               state,r_dev)
                print "opening file %s" % filename
                try:
                    full_buf = read_audio( filename, False )
                except:
                    print " error opening file, skipping!"
                    continue
                print " audio is %d seconds long" % audio_length( full_buf )
                play_time = audio_length( full_buf ) / NUM_PLAY_DEVS

                # split recording into pieces for each playback device
                for p_dev_i in range( NUM_PLAY_DEVS ):
                    recording[user_i][state][r_dev].append([])
                for p_dev_i in range( NUM_PLAY_DEVS ):
                    # cut out portion of wav file
                    buf = audio_window( full_buf, play_time, play_time*p_dev_i)
                    # trim buffer to account for lack of synchrony in play/rec
                    PADDING = 5 # 5 seconds of padding
                    buf = audio_window( buf, play_time-2*PADDING, PADDING) 

                    # now process this buffer with processing_func
                    results = processing_func( buf, processing_func_args)

                    p_dev = int( p_dev_mapping[user_i][p_dev_i] )
                    recording[user_i][state][r_dev][ p_dev ] = results
    return recording


def mean_of_var( buf, args=[[50],20000] ):
    """processes the recording a list of statistics:
        results[num_divisions][statistic_type]
    args is split into [divisions,frequency]
    divisions is a list with the number of partitions to split each recording
    into when calculating the variance of intensity."""
    divisions = args[0]
    freq = args[1]

    results = []
    for div in divisions:
        results.append( process_recording( buf, freq, div ))
    return results

def usenix09_results():
    a = process_all_recordings( "/home/steve/svn/sonar/data/local_study",
                                mean_of_var, [[50],20000] )
    return array( a )

def ubicomp09_results():
    """gives the power in the 20khz spectrum channel

    @return a[user,state,mic,speaker,sample]"""
    a = process_all_recordings( "/home/steve/svn/sonar/data/local_study",
                                bartlett_recording, [20000,500,10] )
    return array( a )

def ubicomp09_plot( a ):
    """calculates the mean of abs(deltas)
    
    @param a[user,state,sample]
    @return deltas[user,state]"""
    diffs = a[:,:,1:] - a[:,:,0:a.shape[2]-1]
    return abs(diffs).mean(axis=2)

def ubicomp09_errorbars( a, divisions=10 ):
    """
    @param a[user,state,sample]
    @return a[user,state,avg/min/max]"""
    window_results = zeros( (divisions,a.shape[0],a.shape[1]) )
    errorbars = zeros( (a.shape[0],a.shape[1],3) )
    samples = a.shape[2]
    samples_per_division = samples/divisions
    for i in range(divisions):
        window = a[:,:, i*samples_per_division:(i+1)*samples_per_division ]
        window_results[i] = ubicomp09_plot( window )
    errorbars[:,:,0] = window_results.mean(axis=0)
    errorbars[:,:,1] = window_results.min(axis=0)
    errorbars[:,:,2] = window_results.max(axis=0)
    return errorbars

def ubicomp09_hw_analysis( a ):
    """calculates, for all mic/speaker combinations the mean ration of video
    echo delta to absent echo delta.

    @return ratio[4,4]"""
    
    b = zeros((4,4,20,5))
    for i in range(4):
        for j in range(4):
            b[i,j] = ubicomp09_plot( a[:,:,i,j] )
    return ( b[:,:,:,1] / b[:,:,:,4] ).mean(axis=2)
            
def ubicomp09_conf_matrix( a, divisions=5 ):
    """Calculates a confusion matrix for a simple kind of video/absent state
    classifier which simply takes the geometric mean of the training absent and
    video state values as a threshold

    @param a[user,state,sample]
    @return a[actual_present(0/1),predicted_present(0/1)]"""
    num_users = a.shape[0]
    num_states = a.shape[1]
    num_samples = a.shape[2]
    samples_per_division = num_samples/divisions
    window_echodelta = zeros( (divisions,num_users,num_states) )

    #===== calculate echo delta
    for i in range(divisions):
        window = a[:,:, i*samples_per_division:(i+1)*samples_per_division ]
        window_echodelta[i] = ubicomp09_plot( window )

    #===== classify
    # initially, record with users separately, trying every pair of windows as training
    conf_matrix = zeros((num_users,divisions,divisions,2,2)) 
    # using each pair of windows as a training set
    video = window_echodelta[:,:,1]
    absent = window_echodelta[:,:,4]
    for i in range(divisions):
        for j in range(divisions):
            # threshold is geometric mean of two training values
            threshold = ( video[i] * absent[j] * absent[j] )**0.3333
                                               # sum over all windows
            conf_matrix[:,i,j,0,0] = ( absent < threshold ).mean(axis=0) 
            conf_matrix[:,i,j,0,1] = ( absent >= threshold ).mean(axis=0)
            conf_matrix[:,i,j,1,0] = ( video < threshold ).mean(axis=0)
            conf_matrix[:,i,j,1,1] = ( video >= threshold ).mean(axis=0)

    # average over all training sets and users
    return conf_matrix.mean(axis=2).mean(axis=1).mean(axis=0)


# for usenix paper data used divisions=50, out3.dat has divisions=[5,50,500]
def write_data( arr, stat=1, div_index=1,
                DIR = "/home/steve/svn/sonar/data/local_study/processed/" ):
    """This is useful only for the array returned by usenix2009_results"""
    for rec_dev in [0,1,2,3]:
        for play_dev in [0,1,2,3]:
            f = open( "%s/%d_%d.%d.txt" % (DIR,rec_dev,play_dev,stat), "w" )
            s = array2string( arr[:,:,rec_dev,play_dev,div_index,stat] )
            f.write("%s\n"%s)
            f.close()


def correlation_helper( buf, args=[20000]):
    """records value for first and second halves"""
    freq = args[0]

    # break into two halves
    half_time = audio_length( buf ) / 2
    divisions = int( round( half_time ) )
    first_half = audio_window( buf, half_time, 0 )
    second_half = audio_window( buf, half_time, half_time )

    r1 = process_recording( first_half, freq, divisions )[1] # [1]=mean_of_var
    r2 = process_recording( second_half, freq, divisions )[1]
    return [r1,r2];


def correlation(directory="/home/steve/svn/sonar/data/local_study"):
    """correlation (among users) between first and second half of recordings.
    Specifically, we are calculating the 
    'Pearson product-moment correlation coefficient' aka 'sample corr. coef.'
    http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    """
    a = array( process_all_recordings(directory,
                                      correlation_helper, [20000] ) )
    corr = zeros( (NUM_STATES, NUM_REC_DEVS, NUM_PLAY_DEVS) )
    num_users = a.shape[0]
    for state in range( NUM_STATES ):
        for rec_dev in range( NUM_REC_DEVS ):
            for play_dev in range( NUM_PLAY_DEVS ):
                # calculate correlation among first and second half of 
                # recordings for all users in this configuration
                #  x = first half, y = second half
                mean_x = a[:,state,rec_dev,play_dev,0].mean()
                mean_y = a[:,state,rec_dev,play_dev,1].mean()
                stddev_x = a[:,state,rec_dev,play_dev,0].std()
                stddev_y = a[:,state,rec_dev,play_dev,1].std()
                sum = 0
                for user in range( num_users ):
                    x_i = a[user,state,rec_dev,play_dev,0]
                    y_i = a[user,state,rec_dev,play_dev,1]                    
                    score_x = ( x_i - mean_x ) / stddev_x
                    score_y = ( y_i - mean_y ) / stddev_y
                    sum += score_x * score_y
                corr[state,rec_dev,play_dev] = sum / float( num_users )
                # NOTE, above, it may be better to use num_users-1 and also
                # recalculate std_dev using n-1 rather than n in denominator
    return [a,corr]


def small_scale_var( buf, args=[50,10,20000] ):
    divisions = args[0]
    subdivisions = args[1]
    freq = args[2]

    window_length = audio_length(buf) / divisions
    energies = []
    for i in range( divisions ):
        buf_window  = audio_window( buf, window_length, i * window_length ) 
        # calculate the energies in each subwindow using bartlett's method
        energies.append( bartlett( buf_window,freq,subdivisions )[0] )
    return energies

def plot_small_scale_var( arr, readings_per_second=10 ):
    """arr must be an array of arr[user,state,readings]"""
    import pylab    
    num_readings = arr.shape[2]
    num_users = arr.shape[0]
    num_states = arr.shape[1]
    
    for i in range( num_states ):
        #pylab.subplot(111, aspect=0.7) # change aspect ratio
        for u in range( num_users ):
            baseline = num_states*u*ones(num_readings)
            timeline = range(num_readings)/(1.0*readings_per_second*ones(num_readings))
            pylab.plot( timeline, baseline + arr[u,i,:], 'k' )
        pylab.xlabel( 'time (seconds)' )
        pylab.ylabel( 'power at 20kHz (offset for each user)' )
        pylab.title( "%s state"% ((['typing','video','phone','puzzle','absent'])[i]))
        filename = "var_fig_%01d.png"%i
        pylab.savefig( filename, dpi=300 ) # second parm is DPI
        pylab.close()


if __name__ == "__main__": local_user_study()
