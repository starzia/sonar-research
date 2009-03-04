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
    """calculates some statistical properties of the passed array.
    This operation is useful as a preprocessing step for ML tools."""
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
        pass


def svm_light_format( training_data, in_file ):
    """A helper function that creates the input files for svm_light"""
    f = open( in_file, "w" )
    for i,str in enumerate( ["+1", "-1"] ):
        # writ data sample vectors
        for sample in training_data[i]:
            f.write( "%s " % str )
            for k in range( sample.size ):
                f.write( "%d:%f " % (k+1, sample[k]) )
            f.write("\n")
    f.close()
    

def svm_train( training_data,
               model_file="/tmp/svm.model", in_file="/tmp/svm.in" ):
    """data[ pos/neg ][ sample_index, data ]
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
    #print "classification success for training was %d percent" %(100.0*result)
    return True


def svm_test( test_data, model_file="/tmp/svm.model", in_file="/tmp/svm.in",
              out_file="/tmp/svm.out" ):
    """data[ pos/neg ][ sample_index, data ]
    returns the accuracy.
    model_file is an input and in_file is a debugging output."""
    from subprocess import Popen,PIPE
    import re
    from threading import Timer
    from string import rstrip

    # test classification model
    svm_light_format( test_data, in_file )
    p= Popen("svm_classify %s %s %s" %(in_file,model_file,out_file),
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
        # remove trailing newline
        for r in range(len(results)): results[r] = float(rstrip( results[r] ))
        return [ accuracy, array(results) ]
    else:
        return [ -0.000001, [] ] # store nonsense value if svm failed


# DEFINE PARAMETERS :
#####################
# training_frac is fraction of data to use as training
training_frac=0.5
# training_slice is which slice of size training_frac to use for
# training.  Eg, 0 means to use the first portion of data, 1 means to use
# the second, etc.
training_slice=0

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
    u = 0 # TODO: hard-code user for now
    quality = zeros( all_params_dims )
    for i in range( quality.size ):
        model_params_i = unravel_index( i, all_params_dims )
        print "testing parameters: %s" % [model_params_i]
        confusion_matrix = test_classifier( model_params_i, data[u] )
        print "confusion matrix:"
        print confusion_matrix
        print
        quality[model_params_i] = eval_conf_matrix( confusion_matrix )

    best_params = quality.argmax()
    # print best conf matrix
    print test_classifier( best_params, data[u] )


def test_classifier( model_params, data ):
    """returns a confusion matrix"""
    #--- preprocess
    [ training_data, test_data ] = preprocess( data, model_params )
    #--- break into positive and negative examples for each state classifier
    training_vectors = make_pos_neg_vectors( training_data )
    #--- build models
    model_filenames = []
    for s in range( len(states) ):
        model_filenames.append( "/tmp/svm_model%d"%s )
        svm_train( training_vectors[s], model_filenames[s] )
    #--- run test vectors through each model
    num_vectors_per_state = samples[ model_params[1] ] * (1-training_frac)
    results = zeros( [ len(states), len(states), num_vectors_per_state ] )
    for s_model in range( len(states) ):
        for s_actual in range( len(states) ):
            # the svm model assigns a numeric value \in [-1,1] to each vector
            results[s_model,s_actual] = svm_test( [test_data[:,s_actual,:],[]],
                                                  model_filenames[s_model] )[1]

    #--- classify, ie choose the best model for each vector
    classification = results.argmax( axis=0 ) # dim: class[s_actual,vec_i]
    confusion_matrix = zeros( [ len(states), len(states) ] )
    for s_model in range( len(states) ):
        confusion_matrix[s_model] = ( classification == s_model ).sum(axis=1)
    confusion_matrix /= num_vectors_per_state # normalize to [0,1]
    return confusion_matrix


def eval_conf_matrix( confusion_matrix ):
    """returns a quality metric for a given confusion matrix"""
    dim = confusion_matrix.shape[0]
    quality = 0
    for i in range( dim ):
        for j in range( dim ):
            if( i == j ):
                # reward true positives
                quality += confusion_matrix[i,j]
    return quality


def preprocess( data, model_params ):
    """return a list of classification vectors based on the passed parameters
    for both training and testing sets: t*_data[sample#,state,feature_index]
    
    data[state,time] is a numpy array"""
    from numpy.fft import fft
    [dm,spl,scl,m] = model_params

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

    #---- add statistics to end of vector
    # I beleive that these are properly called "kernel tricks"
    num_stats = len( classification_stats( new_data[0,0] ) )
    stats = zeros( [ samples[spl], len(states), num_stats ] )
    for i in range(spl):
        for j in range(len(states)):
            stats[i,j] = classification_stats( new_data[i,j] )
    new_data = concatenate( [new_data, stats], 2 ) # axis=2

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
        test_data = vstack( [test1_data, test2_data] )
    # dims are t*_data[sample#,state,feature_index]
    return [ training_data, test_data ]
    

def make_pos_neg_vectors( data ):
    """return ret[ state ][pos/neg ][ sample_index, data ]
    arg data[ sample_index, state, data ]"""
    ret = []
    for s in range(len(states)):
        ret.append([])
        # for this state, label data as either positive or negative examples:
        pos_examples = data[:,s]   # positive examples
        neg_examples = []
        for i in range( len(states) ): # negative examples, from all others
            if( i != s ):
                neg_examples.append( data[:,i] )
        neg_examples = vstack( neg_examples )
        # dims are: *_examples[ sample#, data ]
        ret[s] = [ pos_examples, neg_examples ]
    return ret



def OLD_analyze_param_study( per_state_acc ):
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
                   
