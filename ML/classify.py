#!/usr/bin/python
#-----------------------------------------
# Stephen Tarzia, starzia@northwestern.edu
#-----------------------------------------

from numpy import *

def read_txt_file( filename ):
    """reads a file which has a single value on each line and returns
    as a list."""
    f = open( filename, 'r' )
    elements = f.readlines()
    # remove whitespace and cast as float
    for i in range(length(elements)):
        elements[i] = float(elements.rstrip().lstrip())
    return elements


def classification_stats( elements ):
    """calculates some statistical properties of the
    passed element list.  This operation
    is useful as a preprocessing step for ML tools."""
    new_elements = []
    arr = array( elements )
    stats = []
    stats.append( arr.mean() )
#    stats.append( arr.median() )
    stats.append( arr.var() )
    stats.append( arr.std() )
    stats.append( arr.min() )
    stats.append( arr.max() )
    stats.append( arr.argmin() )
    stats.append( arr.argmax() )
    return stats


def svm( data, model_file="/tmp/svm.model", log_file="/tmp/svm.log", in_file="/tmp/svm.in" ):
    """data[ pos/neg, sample_index, data ]
    returns the accuracy """
    from subprocess import Popen,PIPE
    import re

    # prepare input file
    f = open( in_file, "w" )
    for i,str in enumerate( ["+1", "-1"] ):
        # writ data sample vectors
        for sample in data[i]:
            f.write( "%s " % str )
            for k in range( sample.size ):
                f.write( "%d:%f " % (k+1, sample[k]) )
            # add statistics to end of vector
            stats = classification_stats( sample )
            for k,val in enumerate( stats ):
                f.write( "%d:%f " % (sample.size+k+1, val) )
            f.write("\n")
    f.close()

    # run tool
    p = Popen("svm_learn %s %s" %(in_file,model_file), shell=True, stdout=PIPE)
    output = p.communicate()[0]
    #sts = os.waitpid(p.pid, 0)
    # parse out results
    match = re.search( "\((.*) misclassified,", output )
    misclassified = float( match.group(1) )
    result = misclassified / ( 2.0 * data.shape[1] )
    print "classification success was %f percent" % (100.0*result)
    return result


def classification_param_study( data ):
    """data[user,state,sample] is a numpy array"""
    from numpy.fft import fft
    
    #PARAMETERS :
    # we will run clasification for each user plus for a combination of all:
    users = range( data.shape[0] )
    users.append( "all-users" )
    # we break each user's 50 seconds of samples into this many:
    samples = [1,10,100]
    # the data is modified by these mathematical operations:
    ###scalers = ["none","log","exp","square","sqrt"]
    scalers = ["none"]
    # we look at both the time-domain sample sequence and a freq-domain rep.:
    domains = ["time", "freq"]
    # we will be comparing each pair of states, because the classification
    # problem is binary:
    states_a = ["typing","video","phone","puzzle","absent"]
    states_b = ["typing","video","phone","puzzle","absent"]
    # several Machine Learning approaches are tried:
    ##methods = ["svm","neural net"]
    methods = ["svm"]

    # our figure of merit if the accuracy of the derived clasifier, which will
    # be recorded for each of the combinations of the above parameters:
    accuracy = zeros( [ len(users), len(samples), len(scalers),
                        len(domains), len(states_a), len(states_b),
                        len(methods) ] )

    # For each of the combinations of parameters we will generate a list of
    # classification vectors
    for u,user in enumerate(users):
        for s_a,state_a in enumerate(states_a):
            # we only compare state a to state numbers higher than it
            # to eliminate redundancy:
            for s_b in range( s_a+1, len(states_b) ):
                for j,scaler in enumerate(scalers):
                    # copy data
                    if( user == "all_users" ):
                        # concatenate data from all users
                        scaled_data = empty( [ 2, data.shape[2]*data.shape[0]])
                        scaled_data[0] = data[:,s_a].flatten()
                        scaled_data[1] = data[:,s_b].flatten()
                    else:
                        scaled_data = empty( [ 2, data.shape[2] ] )
                        scaled_data[0] = data[u,s_a]
                        scaled_data[1] = data[u,s_b]
                    # apply the scaler:
                    if( scaler == "log" ):
                        scaled_data = log( scaled_data )
                    elif( scaler == "exp" ):
                        scaled_data = exp( scaled_data )
                    elif( scaler == "square" ):
                        scaled_data = scaled_data**2
                    elif( scaler == "sqrt" ):
                        scaled_data = scaled_data**0.5

                    for m,sample in enumerate(samples):
                        # break the data into the given number of samples:
                        divided_data = array( array_split( scaled_data, sample, axis=1 ) )
                        divided_data = divided_data.swapaxes( 0,1 )
                        for k,domain in enumerate(domains):
                            # if we are looking in frequency domain, apply fft:
                            if( domain == "freq" ):
                                divided_data = fft( divided_data )
                                
                            for l,method in enumerate(methods):
                                print "user=%s states=(%s,%s) scaler=%s samples=%03d\n domain=%s method=%s" % (user,state_a,states_b[s_b],scaler,sample,domain,method) 
                                if( method == "svm" ):
                                    acc = svm( divided_data )
                                elif( method == "neural net" ):
                                    #acc = weka( divided_data )
                                    acc = 0.3
                                accuracy[u,m,j,k,s_a,s_b,l] = acc
                                print
    return accuracy


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "usage is:\n  %s [pickled_data_array_filename]\n"%sys.argv[0]
        sys.exit(-1)
    else:
        arr = load( sys.argv[1] )
        ret = classification_param_study( arr )
                   
