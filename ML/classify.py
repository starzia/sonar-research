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
    # weight positive examples by a factor so that pos and neg accuracy
    # have the same cumulative weight when training
    pos_weight = len(training_data[1]) / float(len(training_data[0]))
    p = Popen("svm_learn -j %f %s %s" % (pos_weight,in_file,model_file),
              shell=True, stdout=PIPE)
    # kill the process if no results in 5 seconds
    Timer( 5, my_kill, [p.pid] ).start()
    output = p.communicate()[0]
    # parse out results
    match = re.search( "\((.*) misclassified,", output )
    if match:
        misclassified = float( match.group(1) )
        result = 1 - ( misclassified / ( len( training_data[0] ) +
                                         len( training_data[1] )    ) )
    else:
        return False
    print "classification success for training was %d percent" % (100.0*result)
    return True


def svm_test( test_data, model_file="/tmp/svm.model", in_file="/tmp/svm.in",
              out_file="/tmp/svm.out" ):
    """data[ pos/neg, sample_index, data ]
    returns the accuracy.
    model_file is an input and in_file is a debugging output."""
    from subprocess import Popen,PIPE
    import re
    from threading import Timer

    # test classification model
    svm_light_format( test_data, in_file )
    p= Popen("svm_classify %s %s" %(in_file,model_file,out_file),
             shell=True,stdout=PIPE)
    # kill the process if no results in 5 seconds
    Timer( 5, my_kill, [p.pid] ).start()
    output = p.communicate()[0]
    # parse out results
    match = re.search( "Accuracy on test set: (.*)%", output )
    if match:
        accuracy = float( match.group(1) ) / 100.0
        f = open( out_file, 'r' )
        results = f.readlines() # this will be a float for each test vec
        f.close()
        return [ accuracy, results ]
    else:
        return [ -0.000001, [] ] # store nonsense value if svm failed


# DEFINE PARAMETERS :
#####################
# we will run clasification for each user plus for a combination of all:
users = range( 20 )
users.append( "all-users" )
# we will be comparing each state to all other states
states = ["typing","video","phone","puzzle","absent"]
# we look at both the time-domain sample sequence and a freq-domain rep.:
domains = ["time", "freq"]
# we break the time-series data into this many samples (windows):
samples = [10,100]
# the data is modified by these mathematical operations:
scalers = ["none","log","exp","square","sqrt"]
# several Machine Learning approaches are tried:
##methods = ["svm","neural net"]
methods = ["svm"]

all_params = [ domains, samples, scalers, methods ]
all_params_dims = []
for i in all_params: all_params_dims.append( len(i) )


def flatten_users( data ):
    """data[user,state,time] is a numpy array
    combines data from all users into one super-user"""
    # collapse users axis:
    shape = data.shape
    new_data = data.reshape( ( 1, shape[1], shape[2]*shape[0] ) )
    return new_data


def classification_param_study( data ):
    """data[user,state,time] is a numpy array"""
    accuracy = eval_param_study( data )
    model_params = analyze_param_study( accuracy )
    
    # now test results using the best one-versus-all classifiers together
    confusion_matrix = test_classifier( model_params, data )


def test_classifier( model_params, data ):
    


def eval_param_study( data,
                      training_frac = 0.5, training_slice=0 ):
    """data[user,state,time] is a numpy array
    training_frac is fraction of data to use as training
    training_slice is which slice of size training_frac to use for
    training.  Eg, 0 means to use the first portion of data, 1 means to use
    the second, etc."""
    from numpy.fft import fft
    
    # our figure of merit if the accuracy of the derived clasifier, which will
    # be recorded for each of the combinations of the above parameters:
    params = zeros( [ len(users), len(states) ].append( all_params.dims ) )
    per_state_acc = zeros( params.shape.append( len(states) ) )

    # EVALUATION :
    ##############
    # For each of the combinations of parameters we will generate a list of
    # classification vectors (samples)
    for i in range( params.size ):
        [u,s,dm,spl,scl,m] = unravel_index( i, params.shape )

        if( users[u] == "all-users" ):
            new_data = flatten_users( data )
            u = 0 # now there is only one "super-user" with index 0
            spl *= len(users)-1 # number of samples is multiplied by # users
        # grab data from one user:
        new_data = new_data[u]
        # dims are: new_data[ state, time ]

        #---- break the data into samples
        # break the data (along time axis) into given number of samples:
        new_data = array( array_split( data, samples[spl], axis=1 ) )
        # dims are new_data[sample#,state,time]
        
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
        # training data may be taken from middle, so we have to split the data
        # and rejoin the left and right test segments
        split1_index = ceil( training_frac * training_slice * samples[spl] )
        split2_index = ceil( training_frac *(training_slice+1) * samples[spl] )
        [test1_data,training_data,test2_data] = array_split( new_data,
                                            [split1_index,split2_index],axis=0)
        if test1_data.size == 0:
            test_data = test2_data
        elif test2_data.size == 0:
            test_data = test1_data
        else:
            test_data = vstack( [test1_data,test2_data] )

        #---- label training_data as either positive or negative examples:
        pos_examples = training_data[:,s]
        neg_examples = []
        for i in range( len(states) ):
            if( i != s ):
                neg_examples.append( training_data[:,i] )
        neg_examples = vstack( neg_examples )
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
                        print " positive test accuracy: ",
                    else:
                        test_data2 = [ [], test_data[:,i] ] # all neg
                        print " negative test %s accuracy: "%states[i],
                    state_acc = svm_test( test_data2 )[0]
                    print state_acc
                    per_state_acc[u,s,dm,spl,scl,m,i] = state_acc
            else:
                "noop. we get here is training fails."
        elif( methods[m] == "neural net" ):
            "noop"
        print
    return per_state_acc


def analyze_param_study( per_state_acc ):
    """per_state_acc[user,state,domain,#samples,scaler,method,state_tested]
    returns the parameters for the best models for each of the states"""
    # ANALYZE RESULTS:
    ##################
    # summarize the accuracies of classifying each state using a weighted
    # average accross states.
    accuracy = per_state_acc.sum(axis=6) # sum all tested states
    # give the positive-state samples equal total weight as negatives:
    for s in range(len(states)):
        accuracy[:,s] += (len(states)-1) * per_state_acc[:,s,:,:,:,:,s]

    # We consider two cases:
    # a) a model is trained for each user, using the first part of their data
    #  In this case, we average the accuracy accross all users for each param
    #  set.
    # b) a single model is trained using the first part of all users' data
    num_users = len(users) - 1
    acc = [ accuracy[num_users],
            accuracy[0:num_users-1].mean(axis=0) ] # avg accross users
    PSacc = [ per_state_acc[num_users],
              per_state_acc[0:num_users-1].mean(axis=0) ] # avg accross users

    # print state-matrix best results
    inter_state_accuracy = zeros( [2,len(states),len(states)] )
    for i in [0,1]:
        model_params = []
        for s in range(len(states)):
            argmax = unravel_index( acc[i][s].argmax(), acc[i][s].shape )
            model_params.append( argmax )
            inter_state_accuracy[i][s] = PSacc[i][s][argmax]
            print "state %d, best_params=%s" % (s,argmax)
        if i:
            print "per-user-model ",
        else:
            print "single-model ",
        print "classification results for each state's best classifier:"
        print " Columns are the actual state."
        print " Rows show the frequency that the state was classified as state i"
        print " using state i's best classifier."
        print inter_state_accuracy[i]
        print

    # return best model params (for per-user-model)
    return model_params


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "usage is:\n  %s [pickled_data_array_filename]\n"%sys.argv[0]
        sys.exit(-1)
    else:
        arr = load( sys.argv[1] )
        ret = classification_param_study( arr )
                   
