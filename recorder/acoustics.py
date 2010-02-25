from extras import *

# CTFM constants
CTFM_ping_period=0.03
CTFM_ping_length=60
CTFM_freq_start=20000
CTFM_freq_end=10000

def CTFM_gnuplot( ping_length = 1, ping_period = 0.03, freq_start = 20000,
                  freq_end = 10000, OFFSET=1, HISTORY=50, DOWNSAMPLE=1 ):
    """gives an interactive view of the cross correlations,
    OFFSET can be used to reduce xcorr resolution to speed up display
    HISTORY is the number of plots to display
    DOWNSAMPLE is the downsample factor, eg 4 means four consecutive samples
    are averaged."""
    # set up the plots
    gnuplot = subprocess.Popen(["gnuplot"], shell=True, \
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    print >>gnuplot.stdin, "set terminal x11;"

    ping = lin_sweep_tone( ping_period, freq_start, freq_end )
    # autocorrelation
    ac = cross_correlation_au( ping, ping, OFFSET, int( ping_period*RATE ) )
    recording=0
    
    # initialize null values for first plotting
    cc = []
    for k in range( HISTORY ):
        cc.append( zeros( len(ac)/DOWNSAMPLE ) )

    num_recordings = 0
    while 1:
        # start non-blocking playback
        full_ping = audio_repeat( ping, int(ceil(ping_length/ping_period))  )
        play_audio( full_ping )
    
        # start non-blocking record
        arecord = nonblocking_record_audio( ping_length, 
                                            "out%d.wav"%(num_recordings%2) )
        
        # process previous iteration's recording, if not 1st iteration
        if( num_recordings > 1 ):
            # shift history
            cc.pop(0)
            recording = read_audio( "out%d.wav" % ((num_recordings-1)%2),
                                    False ) #its mono 
            # calculate cross correlation
            reference = None
            #if num_recordings > 2:
            #    reference = cc[len(cc)-1]
            cc.append( downsample( recording_xcorr( recording, full_ping,
                                   ping_period, OFFSET, reference ), DOWNSAMPLE ) )

            # plot it
            print >>gnuplot.stdin, "set title 'CTFM sonar %dHz to %dHz in %f sec (%dHz sample rate)';" % (freq_start, freq_end, ping_period, RATE )
            print >>gnuplot.stdin, "set cbrange [0:1];" # correlation coefs are normallized
            #print >>gnuplot.stdin, "set isosamples %d,%d;" % ( len(ac), HISTORY )
            print >>gnuplot.stdin, "set xlabel 'crosscorrelation lag (range map) (meters)';"
            print >>gnuplot.stdin, "set ylabel 'time past (s)';"
            #print >>gnuplot.stdin, "set palette positive nops_allcF maxcolors 0 gamma 1.5 gray;"
            print >>gnuplot.stdin, "set palette positive color;"

            # below, -0.5 is for pixel center offset and 330 m/s is speed of sound 
            print >>gnuplot.stdin, "plot '-' using ($2*%f):($1*%f-0.5):3 \
              with image title '';\n" % (DOWNSAMPLE*330.0/RATE, ping_length), 
            k=0
            cc_max = cc[HISTORY-1].max() # normalize to the latest cc vector
            for cc_i in cc:
                k+=1
                j=0
                for i in cc_i:
                    j+=1
                    print >>gnuplot.stdin, "%d %d %f" % (k,j,i/cc_max)
                print >>gnuplot.stdin, ""
            print >>gnuplot.stdin, "EOF"

            #raw_input( 'press enter to continue' )
        
        # wait for recording to finish
        arecord.wait()
        num_recordings += 1

def measure_room_response():
    """measures room impulse response using a one minute Maximum Length
    Sequence."""
    print "recording..."
    mls = read_audio( 'mls_48k_60s.wav', False )
    rec = audio2array( recordback( mls ) )
    mls = audio2array( mls )
    print "xcorr..."
    xc = fft_xcorr( rec, mls )
    # the MLS we use is 65535 samples long
    am = argmax( xc )
    return xc[am:am+65535]
    
def room_response_study( storage_directory ):
    """makes a series of room response recordings, querying the user for a 
    description of each room."""
    import os.path
    mls = read_audio( 'mls_48k_60s.wav', False )

    ping = lin_sweep_tone( CTFM_ping_period, CTFM_freq_start, CTFM_freq_end )
    full_ping = audio_repeat( ping, int(ceil(CTFM_ping_length/CTFM_ping_period))  )
    
    # join sixty seconds of silence, then MLS, then CTFM
    sig = tone( 60, 0 ) + mls + full_ping

    while( 1 ):
        print "Press <enter> to make the next recording"
        sys.stdin.readline()

        rec = recordback( sig )
        description = ''
        while( description == '' or os.path.exists( filename ) ):
            description = raw_input( 'enter a description for the recording filename (no spaces please):' )
            filename = "%s/%s.wav" % (storage_directory,description)
        write_audio( rec, filename, RATE, False )
    

def proc_room_response_study( wav_directory, output_directory ):
    """process the recordings made by room_response_study()"""
    import os
    for dir in [wav_directory, output_directory]:
        if( not os.path.isdir(dir) ):
            print "directory %s does not exist!" % dir
            return
    mls_orig = audio2array( read_audio( 'mls_48k_60s.wav', False ) )
    ping = lin_sweep_tone( CTFM_ping_period, CTFM_freq_start, CTFM_freq_end )
    full_ping = audio_repeat( ping, int(ceil(CTFM_ping_length/CTFM_ping_period))  )
    
    # we will build a list of analysis results
    ir = []
    spec = []
    range_map = []
    names = []

    # read each file in directory
    wav_files = sort( os.listdir(wav_directory) ) # order lexicographically
    for file in wav_files:
        # verify this is a wav file
        name_parts = os.path.splitext( file )
        if( name_parts[1] != '.wav' ):
            continue
        names.append( name_parts[0] )
        basename = "%s/%s"%(output_directory,name_parts[0])
        print 'processing', basename
        buf = audio2array( read_audio( "%s/%s"%(wav_directory,file), False ) )
        
        # split into three one-minute arrays
        silence = buf[0       :RATE*60-1]
        mls     = buf[RATE*60 :RATE*120-1]
        ctfm    = buf[RATE*120:RATE*180-1]

        # get impulse response from mls
        xc = fft_xcorr( mls, mls_orig )
        # the MLS we use is 65535 samples long
        am = argmax( xc )
        output = open( "%s.ir.txt" % basename, 'w' )
        xc = xc[am:am+65535]
        for i in xc:
            print >>output, i
        output.close()
        ir.append( xc )

        # get response spectrum from mls
        [F,Y] = welch_energy_spectrum( xc.tostring() ) 
        output = open( "%s.spec.txt" % basename, 'w' )
        for i in range(len(F)):
            print >>output, "%f\t%f" % ( F[i], Y[i])
        output.close()
        spec.append( Y )

        # get range map from ctfm
        rm = recording_xcorr( ctfm.tostring(), full_ping,
                              CTFM_ping_period )
        output = open( "%s.map.txt" % basename, 'w' )
        for i in rm:
            print >>output, i
        output.close()
        range_map.append( rm )

    # similarity measurements for each pair of recordings, for each analysis type
    print 'computing similarity...'
    N = len( ir )
    similarity = zeros( [3,N,N] ) # 1st dim is analysis type
    output = open( "%s/similarity.txt" % output_directory, 'w' )
    for i in range( N ):
        for j in range( N ):
            if j < i:
                similarity[:,i,j] = similarity[:,j,i] # by symmetry
            else:
                similarity[0,i,j] = dot( ir[i], ir[j] )
                similarity[1,i,j] = dot( spec[i], spec[j] )
                similarity[2,i,j] = dot( range_map[i], range_map[j] )
    # normalize to self-similarity, along column
    for i in range( N ):
        for j in range( N ):
            if i != j:
                similarity[:,i,j] /= similarity[:,i,i]
        similarity[:,i,i] = array([1,1,1])
    # save to file
    for i in range( N ):
        for j in range( N ):
            print >>output, i, j, similarity[0,i,j], similarity[1,i,j], similarity[2,i,j], names[i], names[j]

    return similarity

