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


def my_kill( pid ):
    from os import kill
    from signal import SIGTERM
    try:
        kill( pid, SIGTERM )
    except:
        "noop"


def svm_light_format( training_data, in_file ):
    f = open( in_file, "w" )
    for i,str in enumerate( ["+1", "-1"] ):
        # writ data sample vectors
        for sample in training_data[i]:
            f.write( "%s " % str )
            for k in range( sample.size ):
                f.write( "%d:%f " % (k+1, sample[k]) )
            # add statistics to end of vector
            stats = classification_stats( sample )
            for k,val in enumerate( stats ):
                f.write( "%d:%f " % (sample.size+k+1, val) )
            f.write("\n")
    f.close()
    

def svm( training_data, test_data, model_file="/tmp/svm.model", log_file="/tmp/svm.log", in_file="/tmp/svm.in" ):
    """data[ pos/neg, sample_index, data ]
    returns the accuracy """
    from subprocess import Popen,PIPE
    import re
    from threading import Timer

    # build classification model
    svm_light_format( training_data, in_file )
    p = Popen("svm_learn %s %s" %(in_file,model_file), shell=True, stdout=PIPE)
    # kill the process if no results in 5 seconds
    Timer( 5, my_kill, [p.pid] ).start()
    output = p.communicate()[0]
    # parse out results
    match = re.search( "\((.*) misclassified,", output )
    if match:
        misclassified = float( match.group(1) )
        # below, divide by 2 b/c we have an equal number of negative examples
        result = 1 - ( misclassified / ( 2.0*training_data.shape[1] ) )
    else:
        return -0.000001 # store nonsense value if svm failed
    print "classification success for training was %d percent" % (100.0*result)

    # test classification model
    svm_light_format( test_data, in_file )
    p = Popen("svm_classify %s %s" %(in_file,model_file),shell=True,stdout=PIPE)
    # kill the process if no results in 5 seconds
    Timer( 5, my_kill, [p.pid] ).start()
    output = p.communicate()[0]
    # parse out results
    match = re.search( "Accuracy on test set: (.*)%", output )
    if match:
        accuracy = float( match.group(1) ) / 100.0
        # below, divide by 2 b/c we have an equal number of negative examples
    else:
        accuracy = -1 # store nonsense value if svm failed
    print "classification success for model on new data was %d percent" % (100.0*accuracy)
    return result


def classification_param_study( data ):
    """data[user,state,sample] is a numpy array"""
    from numpy.fft import fft
    
    #PARAMETERS :
    training_frac = 0.2 # fraction of data to use as training
    # we will run clasification for each user plus for a combination of all:
    users = range( data.shape[0] )
    users.append( "all-users" )
    # we break each user's 50 seconds of samples into this many:
    samples = [10,100]
    # the data is modified by these mathematical operations:
    ##scalers = ["none","log","exp","square","sqrt"]
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
                    for m,sample in enumerate(samples):
                        # prepare data
                        if( user == "all-users" ):
                            # concatenate data from all users
                            scaled_data = array( [ data[:,s_a], data[:,s_b] ] )
                            # dimensions are: scaled_data[ pos/neg, user, data]

                            # break the data into the given number of samples:
                            divided_data = array( array_split( scaled_data,
                                                             sample, axis=2 ) )
                            # dims are divided_data[sample#,pos/neg,user,data]
                            # swap axes to put sample# before user so that when
                            # we cut off the first fraction for training this
                            # will represent the first samples from all users
                            # rather than all the data from the first users.
                            divided_data = divided_data.swapaxes( 0,1 )
                            # dims are divided_data[pos/neg,sample#,user,data]
                            shape = divided_data.shape
                            divided_data = divided_data.reshape( 2,
                                                     sample*shape[2], shape[3])
                            # dims are: divided_data[ pos/neg, sample#, data ]
                        else:
                            scaled_data = array( [ data[u,s_a], data[u,s_b] ] )
                            # dimensions are: scaled_data[ pos/neg, data ]

                            # break the data into the given number of samples:
                            divided_data = array( array_split( scaled_data,
                                                             sample, axis=1 ) )
                            divided_data = divided_data.swapaxes( 0,1 )
                            # dims are: divided_data[ pos/neg, sample#, data ]

                        # apply the scaler:
                        if( scaler == "log" ):
                            scaled_data = log( scaled_data )
                        elif( scaler == "exp" ):
                            scaled_data = exp( scaled_data )
                        elif( scaler == "square" ):
                            scaled_data = scaled_data**2
                        elif( scaler == "sqrt" ):
                            scaled_data = scaled_data**0.5

                        for k,domain in enumerate(domains):
                            # if we are looking in frequency domain, apply fft:
                            if( domain == "freq" ):
                                divided_data = fft( divided_data )
                                
                            for l,method in enumerate(methods):
                                # break into training and test fractions:
                                split_index = ceil( training_frac * sample )
                                [training_data,test_data] = array_split(
                                    divided_data, [split_index], axis=1 )
                                
                                print "user=%s states=(%s,%s) scaler=%s samples=%03d\n domain=%s method=%s" % (user,state_a,states_b[s_b],scaler,sample,domain,method) 
                                if( method == "svm" ):
                                    acc = svm( training_data, test_data )
                                elif( method == "neural net" ):
                                    #acc = weka( divided_data )
                                    acc = 0.3
                                accuracy[u,m,j,k,s_a,s_b,l] = acc
                                print
    # We consider two cases:
    # a) a model is trained for each user, using the first part of their data
    #  In this case, we average the accuracy accross all users for each param
    #  set.
    # b) a single model is trained using the first part of all users' data
    num_users = len(users) - 1
    all_user_acc = accuracy[num_users]
    per_user_acc = accuracy[0:num_users-1].mean(axis=0) # avg accross users
    print "maximum single-model accuracy of %f at %s"%( all_user_acc.max(),
                           get_indices( all_user_acc, all_user_acc.argmax() ) )
    print "maximum per-user-model accuracy of %f at %s"%( per_user_acc.max(),
                           get_indices( per_user_acc, per_user_acc.argmax() ) )
    return accuracy


def get_indices( arr, i ):
    """get indices of the ith element in array arr"""
    indices = []
    for s in arr.shape:
        indices.append( i / s )
        i = i % s
    return indices


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "usage is:\n  %s [pickled_data_array_filename]\n"%sys.argv[0]
        sys.exit(-1)
    else:
        arr = load( sys.argv[1] )
        ret = classification_param_study( arr )
                   
