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


def zero_crossings( arr ):
    """return the number of zero crossings in the numpy array."""
    signs = ( arr >= 0 )
    # compare array of signs that are offset by one index
    crossings = signs[1:] - signs[0:signs.shape[0]-1]
    return absolute( crossings ).sum()
    

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
    # append the number of mean-value crossings
    stats.append( zero_crossings( arr - arr.mean() ) )
    return stats


def my_kill( pid ):
    from os import kill
    from signal import SIGTERM
    try:
        kill( pid, SIGTERM )
    except:
        "noop"


def svm_light_format( training_data, in_file ):
    """A helper function that creates the input files for svm_light"""
    f = open( in_file, "w" )
    for i,str in enumerate( ["+1", "-1"] ):
        # writ data sample vectors
        for sample in training_data[i]:
            f.write( "%s " % str )
            for k in range( sample.size ):
                f.write( "%d:%f " % (k+1, sample[k]) )
            # add statistics to end of vector
            # I beleive that these are properly called "kernel tricks"
            stats = classification_stats( sample )
            for k,val in enumerate( stats ):
                f.write( "%d:%f " % (sample.size+k+1, val) )
            f.write("\n")
    f.close()
    

def svm_train( training_data,
               model_file="/tmp/svm.model", in_file="/tmp/svm.in" ):
    """data[ pos/neg, sample_index, data ]
    returns success bool.
    model_file is the primary output and in_file is a debugging output."""
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
        result = 1 - ( misclassified / ( training_data[0].size +
                                         training_data[1].size ) )
    else:
        return False
    print "classification success for training was %d percent" % (100.0*result)
    return True


def svm_test( test_data, model_file="/tmp/svm.model", in_file="/tmp/svm.in" ):
    """data[ pos/neg, sample_index, data ]
    returns the accuracy.
    model_file is an input and in_file is a debugging output."""
    from subprocess import Popen,PIPE
    import re
    from threading import Timer

    # test classification model
    svm_light_format( test_data, in_file )
    p= Popen("svm_classify %s %s" %(in_file,model_file),shell=True,stdout=PIPE)
    # kill the process if no results in 5 seconds
    Timer( 5, my_kill, [p.pid] ).start()
    output = p.communicate()[0]
    # parse out results
    match = re.search( "Accuracy on test set: (.*)%", output )
    if match:
        accuracy = float( match.group(1) ) / 100.0
        # below, divide by 2 b/c we have an equal number of negative examples
    else:
        accuracy = -0.000001 # store nonsense value if svm failed
    print "classification success for model on new data was %d percent" % (100.0*accuracy)
    return accuracy


def classification_param_study( data ):
    """data[user,state,time] is a numpy array"""
    from numpy.fft import fft
    
    # DEFINE PARAMETERS :
    #####################
    training_frac = 0.5 # fraction of data to use as training
    # we will run clasification for each user plus for a combination of all:
    users = range( data.shape[0] )
    users.append( "all-users" )
    # we break the time-series data into this many samples (windows):
    samples = [10,100]
    # the data is modified by these mathematical operations:
    scalers = ["none","log","exp","square","sqrt"]
    # we look at both the time-domain sample sequence and a freq-domain rep.:
    domains = ["time", "freq"]
    # we will be comparing each state to all other states
    states = ["typing","video","phone","puzzle","absent"]
    # several Machine Learning approaches are tried:
    ##methods = ["svm","neural net"]
    methods = ["svm"]

    # our figure of merit if the accuracy of the derived clasifier, which will
    # be recorded for each of the combinations of the above parameters:
    accuracy = zeros( [ len(users), len(states),
                        len(domains), len(samples), len(scalers),
                        len(methods) ] )

    # EVALUATION :
    ##############
    # For each of the combinations of parameters we will generate a list of
    # classification vectors (samples)
    for i in range( accuracy.size ):
        [u,s,dm,spl,scl,m] = get_indices( accuracy, i )

        #---- break the data into samples
        # dims are: data[ user, state, time ]
        # break the data (along time axis) into given number of samples:
        new_data = array( array_split( data, samples[spl], axis=2 ) )
        # dims are new_data[sample#,user,state,time]
        
        if( users[u] == "all-users" ):
            # collapse users axis:
            shape = new_data.shape
            new_data = new_data.reshape( shape[0], shape[2], shape[3] )
        else:
            # grab data from one user:
            new_data = new_data[:,u]
        # dims are: new_data[ sample#, state, time ]

        #---- apply the data scaler:
        if( scalers[scl] == "log" ):
            new_data = log( new_data )
        elif( scalers[scl] == "exp" ):
            new_data = exp( new_data )
        elif( scalers[scl] == "square" ):
            new_data = new_data**2
        elif( scalers[scl] == "sqrt" ):
            new_data = new_data**0.5

        #---- apply the domain transform
        if( domains[dm] == "freq" ):
            new_data = fft( new_data )

        #---- break into training and test fractions:
        split_index = ceil( training_frac * samples[spl] )
        [training_data,test_data] = array_split( new_data,[split_index],axis=0)

        #---- label training_data as either positive or negative examples:
        pos_examples = training_data[:,s]
        neg_examples = []
        for i in range( len(states) ):
            if( i != s ):
                neg_examples.append( training_data[:,i] )
        neg_examples = hstack( neg_examples )
        # dims are: *_examples[ sample#, time ]
        training_data = [ pos_examples, neg_examples ]

        #---- now run ML algorithm
        print "user=%s state=%s scaler=%s samples=%03d domain=%s method=%s" % (users[u],states[s],scalers[scl],samples[spl],domains[dm],methods[m]) 
        if( methods[m] == "svm" ):
            if( svm_train( training_data ) ):
                # for all states other than the one we are currently detecting:
                for i in range( len( states ) ):
                    if( i == s ):
                        test_data2 = [ test_data[:,i], [] ] # all pos
                        true_pos_accuracy = svm_test( test_data2 )
                        print " positive test accuracy: %f" % true_pos_accuracy
                    else:
                        test_data2 = [ [], test_data[:,i] ] # all neg
                        true_neg_accuracy = svm_test( test_data2 )
                        print " negative test %s accuracy: %f" % ( states[i],
                                                            true_neg_accuracy )
                acc = 1 # TODO
            else:
                acc = -0.00000000008 #nonsense, error code on failure
        elif( methods[m] == "neural net" ):
            #acc = weka( new_data )
            acc = -0.0000003
        accuracy[u,s,dm,spl,scl,m] = acc
        print

    # ANALYZE RESULTS:
    ##################
    # We consider two cases:
    # a) a model is trained for each user, using the first part of their data
    #  In this case, we average the accuracy accross all users for each param
    #  set.
    # b) a single model is trained using the first part of all users' data
    num_users = len(users) - 1
    all_user_acc = accuracy[num_users]
    per_user_acc = accuracy[0:num_users-1].mean(axis=0) # avg accross users
    print all_user_acc.shape
    print "maximum single-model accuracy of %f at %s"%( all_user_acc.max(),
                           get_indices( all_user_acc, all_user_acc.argmax() ) )
    print "maximum per-user-model accuracy of %f at %s"%( per_user_acc.max(),
                           get_indices( per_user_acc, per_user_acc.argmax() ) )
    print
    # print state-matrix best results
    for i in [0,1]:
        inter_state_accuracy = zeros( [len(states_a),len(states_b)] )
        for s_a in range(len(states_a)):
            for s_b in range(len(states_b)):
                if i:
                    inter_state_accuracy[s_a,s_b] = per_user_acc[s_a,s_b].max()
                else:
                    inter_state_accuracy[s_a,s_b] = all_user_acc[s_a,s_b].max()
        if i:
            print "per-user-model best inter-state classification accuracy:"
        else:
            print "single-model best inter-state classification accuracy:"
        print inter_state_accuracy
        print
    return accuracy


def get_indices( arr, i ):
    """get indices of the ith element in high-dimensional array arr"""
    indices = []
    slice_size = arr.size
    for s in arr.shape:
        slice_size /= s
        indices.append( i / slice_size )
        i %= slice_size
    return indices


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "usage is:\n  %s [pickled_data_array_filename]\n"%sys.argv[0]
        sys.exit(-1)
    else:
        arr = load( sys.argv[1] )
        ret = classification_param_study( arr )
                   
