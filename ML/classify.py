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
    #stats.append( arr.median() )
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
# we will be comparing each state to all other states
states = ["typing","video","phone","puzzle","absent"]
# we look at both the time-domain sample sequence and a freq-domain rep.:
domains = ["time", "freq"]
# we break the time-series data into this many samples (windows):
samples = [10,100]
# the data is modified by these mathematical operations:
scalers = ["none","log","exp","square","sqrt"]
# statistics to include:
# exclusive means *only* use stats in the training vectors 
statistics = ["none","all","exclusive"] 
# several Machine Learning approaches are tried:
##methods = ["svm","neural net"]
methods = ["svm"]

all_params = [ domains, samples, scalers, statistics, methods ]
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
    best_quality = -9999999
    # for all parameter settings
    # compute confusion matrix, and its quality
    for i in range( array(all_params_dims).prod() ):
        model_params_i = unravel_index( i, all_params_dims )
        print "testing parameters: %s" % [model_params_i]
        # confusion matrix will calculated for each user, then totalled
        confusion_matrices = zeros( [ len(users), len(states), len(states) ] )
        for u in users:
            confusion_matrices[u] = classifier_confusion( model_params_i,
                                                          data[u] )
        # total confusion matrix
        confusion_matrix = confusion_matrices.mean(axis=0)
        print "confusion matrix:"
        print confusion_matrix
        quality = eval_conf_matrix( confusion_matrix )
        print "quality = %f" % quality
        print
        # keep track of best so far
        if quality > best_quality:
            best_quality = quality
            best_params = model_params_i
            best_confusion_matrix = confusion_matrix

    # print best confusion matrix
    print "best parameters: %s" % [best_params]
    print "confusion matrix:"
    print best_confusion_matrix
    print "best quality = %f" % best_quality


def run_svm_state_models( model_params, data ):
    """returns results[state_model][state_actual][vector]
    arg data[state,time]"""
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
    return results


def classifier_confusion( model_params, data ):
    """returns a confusion matrix
    arg data[state,time] is a numpy array

    Classification is done by running data agains svm models for each state
    and choosing the state whose model returned the highest confidence value"""
    results = run_svm_state_models( model_params, data )
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
        # reward true positives by simply summing the confusion matrix diagonal
        quality += confusion_matrix[i,i]
    return quality


def preprocess( data, model_params ):
    """return a list of classification vectors based on the passed parameters
    for both training and testing sets: t*_data[sample#,state,feature_index]
    
    data[state,time] is a numpy array"""
    from numpy.fft import fft
    [dm,spl,scl,sts,m] = model_params

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

    #---- add statistics to end of vector, if appropriate
    # I beleive that these are properly called "kernel tricks"
    num_stats = len( classification_stats( new_data[0,0] ) )
    stats = zeros( [ samples[spl], len(states), num_stats ] )
    for i in range( samples[spl] ):
        for j in range(len(states)):
            stats[i,j] = classification_stats( new_data[i,j] )
    if( statistics[sts] == "all" ):
        new_data = concatenate( [new_data, stats], 2 ) # axis=2
    elif( statistics[sts] == "exclusive" ):
        new_data = stats # replace data with statistics

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


def generate_engagement_metric( data, model_params ):
    """arg data[ user, state, time ]"""
    # we use a simple engagement metric which is just the sum of the svm
    # classifier outputs for all of the 5 models.
    results = []
    for u in [0,1]:
        results.append( run_svm_state_models( model_params, data[u] ) )
    results = array(results) # dims: results[user][model_state][actual_state][]
    # sum accross all models to get a single engagement measure for all vectors
    results = results.sum( axis=1 )
    for s in range(len(states)):
        print 
        for i in results[:,s,:].flatten():
            print "%d\t%f" % (s,i)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "usage is:\n  %s [pickled_data_array_filename]\n"%sys.argv[0]
        sys.exit(-1)
    else:
        arr = load( sys.argv[1] )
        ret = classification_param_study( arr )
                   
