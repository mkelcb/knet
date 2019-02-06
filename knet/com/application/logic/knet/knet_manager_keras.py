# -*- coding: utf-8 -*-

#MIT License

#Copyright (c) 2017 Marton Kelemen

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from keras.layers import Flatten, Activation, BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l1_l2
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.advanced_activations import LeakyReLU, PReLU

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

from ....application.utils import plotgen
from ....application.utils import geno_qc
from ....application.logic.knet import knet_main
from ....io import knet_IO
import gc
import numpy as np
from numpy.linalg import norm 
from scipy import stats
from pathlib import Path
import random
import os


# delta = (Ve/Vg)
# delta = (1-h2) / h2
#args, args.epochs, args.learnRate, args.momentum, args.evalFreq, args.savFreq, args.predictPheno, args.loadWeights, args.saveWeights, args.randomSeed, args.hidCount, args.hidl2, args.hidAct
#  V(G)    0.168545        0.004763
#V(e)    0.006826        0.002168



# =============================================================================

# CUSTOM EARLY STOPPING LOGIC: as NNs that have a difficult start will get aborted before they learn antthing
# so we only want early stop AFTER the NN started converging to any degree
import warnings

class Callback(object):
    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

# evaluates the rSQ at given freq intervals
class evalR_SQ(Callback):
    # same as Keras, EarlyStopping, but only kicks in once NN started converging
    # set threshold to something above chance level, to make sure NN isn't stuck at the nonlearned state

    def __init__(self, model, results, M, y, modelEpsilon, M_validation = None, y_validation = None, freq=10):
        super(evalR_SQ, self).__init__()

        self.model = model
        self.modelEpsilon = modelEpsilon

        self.M = M
        self.y = y 
        self.M_validation = M_validation
        self.y_validation = y_validation
        self.results = results
        self.currentEpoch = 0
        self.freq = freq
        
        self.results["epochs"] = list()
        if self.M is not None : self.results["train_accuracy"]  = list()
        if self.M_validation is not None : self.results["test_accuracy"]  = list()
        
        
    def on_epoch_end(self, epoch, logs=None):
        
        if self.currentEpoch % self.freq == 0:
            self.results["epochs"].append(self.currentEpoch) # add the epoch's number
            
            evaluation = "prediction (r^2)"
            resultsText = ""
    
            if self.M is not None:
                yhatKeras = self.model.predict(self.M)
                yhatKeras += self.modelEpsilon # for numerical stability
                rSQ = np.corrcoef( self.y, yhatKeras, rowvar=0)[1,0]**2  # 0.1569    
                self.results["train_accuracy"].append(rSQ) 
                resultsText += "Training " +evaluation +":" +  str(rSQ) + " / "
    
            
            if self.M_validation is not None:
                yhatKeras = self.model.predict(self.M_validation)
                yhatKeras += self.modelEpsilon # for numerical stability
                rSQ = np.corrcoef( self.y_validation, yhatKeras, rowvar=0)[1,0]**2  # 0.1569  
                self.results["test_accuracy"].append(rSQ)
                resultsText += "Test " +evaluation +":" +  str(rSQ)
                
            print(resultsText, flush = True)
            
        self.currentEpoch += 1
  
def addActivation(model, hidAct): 
    if hidAct == 1 :  model.add(Activation('sigmoid'))	
    elif hidAct == 2 :  model.add(Activation('relu'))	
    elif hidAct == 3 :  print("no activatioN")
    elif hidAct == 5 :  model.add(LeakyReLU(alpha =0.001 ))	
    else : print("softplus not implemented") 
    

def build_keras_model(args, modelEpsilon, input_shape, hiddenShrinkage,  M , y , M_validation = None, y_validation = None) :
    hLayerCount = args.hidCount
    k.set_image_dim_ordering('tf') # otherwise it would give 'negative dimension size' for maxpool operations
    #k.image_dim_ordering()
    BNEnabled = int(args.bnorm) == 1

    decay_Enabled = int(args.lr_decay) == 1

    shrinkage = hiddenShrinkage

   # k.set_image_dim_ordering('tf')
    
    tf.set_random_seed(args.randomSeed) ## tf runs off a different random generator, let it be random for now, to be able to see if it really is the random init that is causing the 'zero' results?

    if args.optimizer == 1 : 
        optimizer=Adam(lr=args.learnRate,    epsilon=modelEpsilon, beta_1=args.momentum, beta_2=0.999, 	decay=args.lr_decay)  # for float16, otherwise we get NaNs
    else :
        optimizer = SGD(lr=args.learnRate, momentum=args.momentum, decay=args.lr_decay)


	 # , beta_1=0.99, since the new Batchnorm logic this isn't needed
    loss = 'mean_squared_error'
    accMetrc = 'mae'


    # Set up the base model.
    model = Sequential()
    
    lastLayerSize = args.firstLayerSize #lastLayerSize_MAX
    # Input = knet_main.knnLayer( myNet,np.array([-1]), knet_main.LAYER_SUBTYPE_INPUT)

    w_reg = l1_l2(l1=0.0, l2=hiddenShrinkage)
 #    if conv was enabled we then do NOT regularize stuff at the first FC layer as we only want to regularize by h2 once
    if args.convLayers > 0 :
        lastOutput = input_shape[0] # for conv nets it is channels last format for Tensorflow, so the first element of the input shape is the actual number of SNPs
        print("Adding "+str(args.convLayers)+" conv layers, with initial input dimension: " + str(lastOutput), flush=True)

        currentNumFilters= args.convFilters
        currentStride = 3
        filter_size = 5 # as it turns out it is not actually a problem if the conv outputs something that isn't an integer, so we just need to downsample it
        
        model.add(Conv1D(currentNumFilters, filter_size, input_shape=(input_shape),kernel_regularizer=w_reg, kernel_initializer='he_normal' , padding="same", strides=currentStride ))
        if BNEnabled : model.add(BatchNormalization())    
        addActivation(model,args.hidAct)
        if args.dropout != -1 : model.add(Dropout(args.dropout))
    
    
    # k.floatx() # 'float32'
  #  args.convLayers = 2


        lastOutput = (lastOutput - filter_size +2) / currentStride + 1
        lastOutput = int(lastOutput) # as these can only be integers
        print("filter size : " + str(filter_size), flush=True)
        #i=1
        shrinkage = 0.0
        currentStride = 1
        pool_size = 2
        for i in range(1, args.convLayers +1) :
            
            # decide on filter size, depending on input, Conv layers must always produce even outputs so that maxpool can half them
            filter_size = 3
            if lastOutput % 2 != 0 : filter_size = 4 # if the current output is not even, then we have to use a filter size of 4, otherwise we get fractions after the maxpool operation
            ## currentNumFilters = (i+1) * args.convFilters
            currentNumFilters = currentNumFilters // 2
            
            
            model.add(Conv1D(currentNumFilters, filter_size,kernel_regularizer=None, kernel_initializer='he_normal' , padding="same", strides=currentStride))
            if BNEnabled : model.add(BatchNormalization())    
            addActivation(model,args.hidAct)
            if args.dropout != -1 : model.add(Dropout(args.dropout))
            
            
            lastOutput = (lastOutput - filter_size +2) / currentStride + 1
            lastOutput = int(lastOutput) # as these can only be integers
            print("filter size affter Conv ("+str(i)+") : " + str(filter_size) + " / output: " + str(lastOutput), flush=True)
            
            lastOutput = (lastOutput - pool_size) / pool_size + 1 # compute what dimensions the conv+maxpool operations are going to leave for the next layer
            print("filter size affter Maxpool ("+str(i)+") : " + str(filter_size) + " / output: " + str(lastOutput), flush=True)
            model.add(MaxPooling1D())  # the default is 2
            
        model.add(Flatten()) # Flatten the data for input into the plain hidden layer
        


    for i in range(1,hLayerCount+1) : # iterate 1 based, otherwise we will get a reduction after the first layer, no matter the widthReductionRate, as 0 is divisible by anything
        if i > 1 or args.convLayers > 0 : shrinkageParam = w_reg # only add regularizer for first layer, subsequent layers will always have none  
        else : shrinkageParam = None
        #if i == (hLayerCount-1) : lastWidth = 2 # enforce so that the last widht is always 2, ie 1 neuron makes it MORE like the other LESS likely
    
 
        if i == 1:  model.add(Dense(lastLayerSize,kernel_regularizer=shrinkageParam, kernel_initializer='he_normal', input_shape = input_shape)) 
        else : model.add(Dense(lastLayerSize,kernel_regularizer=shrinkageParam, kernel_initializer='he_normal'))
        
        if BNEnabled : model.add(BatchNormalization())
        addActivation(model,args.hidAct)
        
        if args.dropout != -1 : model.add(Dropout(args.dropout))
        
        print("added layer at depth: " + str(i) + " with width: " + str(lastLayerSize) + " / shrinkage: " + str(shrinkage))
        
        # control the 'fatness' of the network: we reduce the width at a given rate: if this is 1, then at every subsequent layer, if its 2, then every 2nd layer etc
        if i % args.widthReductionRate == 0 :  lastLayerSize = lastLayerSize // 2
        
        if lastLayerSize < 2 : break # if 
        
    #
    if hLayerCount == 0 : model.add(Dense(1, kernel_initializer='he_normal',kernel_regularizer=w_reg, input_shape = input_shape))	
    else : model.add(Dense(1, kernel_initializer='he_normal',kernel_regularizer=shrinkageParam))	
    
    if len( y.shape) > 1 :  model.add(Activation('softmax'))
    
    
    # Compile the model.	
    model.compile(loss=loss, optimizer=optimizer, metrics=[accMetrc])
    
    return(model)


def runKnet(args) :
    print("KNeT via Keras backend")
    hiddenShrinkage = args.hidl2
    # default QC settings used for all non AMBLUP versions
    _minObserved = 0.95
    _minMAF = 0.01
    _minVariance = 0.02
    
    # load plink binary / phenotypes want to load them here, so that we can release the memory once the raw data is no longer used
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : recodecc = False

    
    genotypeData = knet_IO.loadPLINK(args.knet, loadPhenos = False) 
    M = genotypeData["M"]
    irsIds = genotypeData["rsid"]
    IDs = genotypeData["IDs"] 
    indicesKept = np.asarray( range(M.shape[1]) )
    
    del genotypeData ; gc.collect() # dont need this
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
    y = stats.zscore(y) # zscore it so that Beta -> h2 computations work
 
#    M = M[:,0:M.shape[1]//30]
#    M = M[0:M.shape[0]//10]
#    y = y[0:y.shape[0]//10] 
    
    # if we have a validation set
    M_validation = None
    y_validation = None
    if args.validSet :
        genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False, replaceMissing = True) # want to replace -1 with 0s, as we otherwise would have -1s, as later we just delete indices that failed QC for the training set, but won't care for individual missing datas
        M_validation = genotypeData["M"] 
        IDs_validation = genotypeData["IDs"] 
        del genotypeData ; gc.collect() # dont need this
        
        if args.validPhen :
            y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
            y_validation = stats.zscore(y_validation) # zscore it so that Beta -> h2 computations work

 
    if args.inference == 0 : 
        # 1. standardise data
        if args.qc == 1 :
            qc_data = geno_qc.genoQC_all(M, rsIds = irsIds, minObserved = _minObserved, minMAF = _minMAF, minVariance = _minVariance) # we MUST perform QC with the EXACT SAME settings as the 'region scanner' otherwise the region coordinates will be mismatched
            #M = qc_data["X"] 
            rsIds_qc = qc_data["rsIds"] # save away the surviving SNP list that we have used 
            indicesToRemove = qc_data["indicesToRemove"]
            indicesKept = qc_data["indicesKept"]
            irsIds = rsIds_qc.tolist()
            
            del qc_data; gc.collect() # overwrite
            
            qc_data = geno_qc.removeList(M, indicesToRemove)
            M = qc_data["X"]
            del qc_data; gc.collect() # overwrite
        else : print("Skipping internal QC", flush=True)
        
        M, mns, sstd = geno_qc.standardise_Genotypes(M) ; gc.collect()
        print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )

    else :
        print("Inference data QC", flush=True)
        if args.snpIndices is not None :
            indicesToKeep = knet_IO.loadIndices(args.snpIndices)
            M = M[:,indicesToKeep]
            
        mns  = knet_IO.loadVectorFromDisk( args.mns  , 'float32')  # these are always float32 even in 64 runs
        sstd = knet_IO.loadVectorFromDisk( args.sstd , 'float32')  
        snpIDs = knet_IO.loadsnpIDs(args.snpIDs)
        
        

        M = M.astype('float32')
        M -= mns
        M /= sstd
        # load final list of RSids
        # load mean /SDs
       
    #M = geno_qc.standardise_Genotypes(M) ; gc.collect()
    #print("After standardising, training data in MBs is: ",geno_qc.getSizeInMBs(M) )
    

    # get Zscores: have to standardise ONLY over the training, and not the training+ validation together: https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
        # will have to implement this for genetic data
    if M_validation is not None : 
        if args.qc == 1 :
            # depending on if we are in inference mode, make sure we have the same set of SNPs
            if args.inference == 0 : M_validation = np.delete(M_validation, indicesToRemove, axis=1)
            else : M_validation = M_validation[:,indicesToKeep]
            
        #qc_data = geno_qc.removeList(M_validation, indicesToRemove)
        M_validation = M_validation.astype('float32')
        M_validation -= mns
        M_validation /= sstd
        indices_validation = np.asarray( range(len(M_validation)) ) # is used for storting
        print("After standardising, validation data in MBs is: ",geno_qc.getSizeInMBs(M_validation) )
        
        
        
#


    # Pre-process data:
    evalTrainResults = True    


    # Shuffle data before producing the minibatches to avoid having all-case or all-control minibatches
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    indices = np.asarray( range(len(M)) ) # is used for storting
    random.shuffle(indices)
    M = M[indices]
    y = y[indices]
    IDs[0] = np.array(IDs[0])
    IDs[1] = np.array(IDs[1])
    IDs[0] = IDs[0][indices]
    IDs[1] = IDs[1][indices]
    # 2. create minibatch list
    numIndividuals = M.shape[0] 
    numSNPs = M.shape[1] # numSNPs = bed.get_nb_markers(), as we may have removed SNPs, we want to know how many are left
    len_M = len(M)
    len_M_validation = 0
 
    # reshape data to be the right dimensions for Convolutions
    if args.convLayers > 0 :
        input_shape = [numSNPs,1,]
        M = M.reshape(M.shape[0], M.shape[1], 1)  # M = M.reshape(M.shape[0], 1 , 1, M.shape[1])
        if M_validation is not None :
            M_validation = M_validation.reshape(M_validation.shape[0],M_validation.shape[1],  1)  ## M_validation.reshape(M_validation.shape[0], 1 , 1, M_validation.shape[1]) 

    else : input_shape =  (numSNPs,)

    minibatch_size =  args.batch_size #M.shape[0]  # 64   #minibatch_size = 128
#   minibatch_size = 128
    if args.batch_size == 0 : minibatch_size = len(M)
    num_batches = len(M) // minibatch_size

    # scale the delta by minibatch_size, if we dont have minibatches
    ratio = float(minibatch_size) / numIndividuals # this is 1 if there are no minibatches
    print("orig L2 Regularizer : " + str(hiddenShrinkage) + " minibatches scaled to " + str(hiddenShrinkage * ratio) )
    hiddenShrinkage *= ratio


    # 3. initialise network params     
    floatPrecision = "float" +str(args.float) 
    print("floatPrecision is: " + floatPrecision)
    #knet_main.setDataType(floatPrecision)
    modelEpsilon =1e-08

    if args.gpu == 1 : 
        print("attempting to init GPU", flush=True)
        config = tf.ConfigProto()
        with k.tf.device('/gpu:0'):  # ('/gpu:'+ args.gpu_target):
          model = build_keras_model(args, modelEpsilon, input_shape, hiddenShrinkage,  M , y , M_validation = M_validation, y_validation = y_validation)

    else :
        print("Using CPUs only, num cores: " , args.num_CPU)
        model = build_keras_model(args, modelEpsilon, input_shape, hiddenShrinkage,  M , y , M_validation = M_validation, y_validation = y_validation)

        #config.device_count={'CPU': args.num_CPU, 'GPU': 0} # this does not work, fo rsome reason I cannot set the device counts on the config once its created citing some rubbish about "Assignment not allowed to repeated field device_count"
        config = tf.ConfigProto(device_count={'CPU': args.num_CPU, 'GPU': 0})
            





    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    #config.log_device_placement=True
    
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))


    if args.inference == 0 :
        print("Analysis Run", flush = True)
        results_its = {}
        evalCallback = evalR_SQ(model, results_its, M, y, modelEpsilon, M_validation = M_validation, y_validation = y_validation, freq=args.evalFreq)
        allCallbacks = [evalCallback]
        
        if args.saveWeights is not None :
            best_model = ModelCheckpoint(filepath=args.saveWeights, verbose=0, save_best_only=True, save_weights_only=True) # save_weights_only=True: need this otherwise will get a tensorflow bug in the multiGPU case
            allCallbacks.append(best_model)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)  
      
        k.get_session().run(tf.initialize_all_variables()) # initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02. Instructions for updating: Use `tf.global_variables_initializer` instead.
        valData = None
        if M_validation is not None : valData = (M_validation, y_validation)
        model.fit(M, y, validation_data=valData, epochs=args.epochs, batch_size=minibatch_size, callbacks=allCallbacks, verbose=1)

        if M_validation is not None:
            yhatKeras = model.predict(M_validation)
            yhatKeras += modelEpsilon # for numerical stability
            rSQ = np.corrcoef( y_validation, yhatKeras, rowvar=0)[1,0]**2  # 0.1569  
            print("Keras validation rSQ: " + str(rSQ) )
        

        
 
    
        #write training data means / stds to disk so that we could use those for inference runs later
        print("writing means/stds to disk with datatype: "  + str(sstd.dtype))
        print("sstd shape is: " + str(sstd.shape) + " / mns shape: " + str(mns.shape))
        
        knet_IO.writeVectorToDisk( args.out + "data_mns" , mns, mns.dtype)  
        knet_IO.writeVectorToDisk( args.out + "data_sstd" , sstd, sstd.dtype)  
    
           
        fileName = args.out + "nn_results.txt"
        with open(fileName, "w") as file: 
            
            line = "epochs"
            if "train_accuracy" in results_its: line = line + "\t" + "train_accuracy"
            if "test_accuracy" in results_its: line = line + "\t" + "test_accuracy"
            file.write(line  + "\n")
             
            for i in range( len(results_its["epochs"])  ):
                line = str(results_its["epochs"][i]) 
                if "train_accuracy" in results_its: line = line + "\t" + str(results_its["train_accuracy"][i])
                if "test_accuracy" in results_its: line = line + "\t" + str(results_its["test_accuracy"][i])
                file.write(line + "\n")            
            
        
        # generate plot of the results
        if len(results_its["epochs"]) > 0 :
            plotgen.exportNNPlot(results_its, args.out + "nnplot")
        
        
        # write out the SNPs that were used for the analysis
        fileName = args.out + "nn_SNPs.txt"
        with open(fileName, "w") as file: 
            for i in range( len(irsIds)  ):
                file.write(irsIds[i]  + "\n")
        
        # write out the indices of the original dataset's coordinates for convenience
        if indicesKept is not None: # in case we skipped QC
            fileName = args.out + "nn_SNPs_indices.txt"
            with open(fileName, "w") as file: 
                for i in range( len(indicesKept)  ):
                    file.write( str(indicesKept[i])  + "\n")    
             
                
        if len_M_validation > 0 :
            producePRS(model,M_validation, IDs_validation,  args.out + "yhat.txt", args.out + "FIDs.txt", y_validation, args.out + "KNET_PRS")
#            
#            # write final predictions out
#            yhat_all = yhatKeras = model.predict(M_validation)
#
#       
#            fileName = args.out + "yhat.txt"
#            with open(fileName, "w") as file:
#                file.write("Profile"  + "\n")
#                
#        
#                for i in range(yhat_all.shape[0]) :
#                    line = str(yhat_all[i][0] )
#                    for j in range(1, len(yhat_all[i]) ):
#                        line = line + "\t" + str(yhat_all[i][j] )
#                        
#                    file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")
#        
#            # also write out the FID / IIDs in the same order, just as a sanity check (compare this against the .fam files)
#            
#            fileName = args.out + "FIDs.txt"
#            with open(fileName, "w") as file:
#                file.write("FID" + "\t" + "IID" + "\n")
#        
#                for i in range( len(IDs_validation[0]) ) :
#                    line = IDs_validation[0][i] + "\t" + IDs_validation[1][i]
#        
#                    file.write( line +  "\n")  
                    
                    

        k.clear_session() # once model is trained we want to clear it otherwise we ll get memory leaks:  https://stackoverflow.com/questions/42047497/keras-out-of-memory-when-doing-hyper-parameter-grid-search

                 
    else :
        print("Inference Run", flush = True)
        model.load_weights(args.loadWeights) 
        
        
        if len_M_validation > 0 :
            producePRS(model,M_validation, IDs_validation,  args.out + "yhat.txt", args.out + "FIDs.txt", y_validation, args.out + "KNET_PRS")  

        k.clear_session() # once model is trained we want to clear it otherwise we ll get memory leaks:  https://stackoverflow.com/questions/42047497/keras-out-of-memory-when-doing-hyper-parameter-grid-search

       
        
#
#        
#         
#        os.makedirs(os.path.dirname(args.out), exist_ok=True)
#        # forward propagate with the 1st sample of the training set 
#        yhat = myNet.forward_propagate(train_GWAS[0], train = False, saveInput = False, forceCast_toCPU = True)
#        
#        suppressPrint_orig = myNet.suppressPrint
#        myNet.suppressPrint = True
#        StartImage = None
#        #StartImage = np.random.normal( size=(1,X_test.shape[1]))
#        dream = myNet.dream(0, 100,StartImage,200 , mFilterSize = 0, blur = 0.0, l2decay = 0.0, small_norm_percentile = 0,lr = 1.5,normalize = True, small_val_percentile = 0)
#        NNinference = dream[0].ravel()
#        NNinference[np.isnan(NNinference)]=0.0
#        myNet.suppressPrint = suppressPrint_orig 
#        
#            # Here this would need to be more constrained:
#            # both LD and MAF need to be taken into account
#
#        knet_IO.writeSNPeffects(args.out + "dream",snpIDs, NNinference)
    
   
def producePRS(model,M_validation, IndiIDs, outLoc_yhat, outLoc_FIDs,ytrue, outLoc_PRS) :
    # write final predictions out
    yhat_all = model.predict(M_validation)

   
    fileName = outLoc_yhat
    with open(fileName, "w") as file:
        file.write("Profile"  + "\n")
        

        for i in range(yhat_all.shape[0]) :
            line = str(yhat_all[i][0] )
            for j in range(1, len(yhat_all[i]) ):
                line = line + "\t" + str(yhat_all[i][j] )
                
            file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")

    # also write out the FID / IIDs in the same order, just as a sanity check (compare this against the .fam files)
    
    fileName = outLoc_FIDs
    with open(fileName, "w") as file:
        file.write("FID" + "\t" + "IID" + "\n")

        for i in range( len(outLoc_FIDs[0]) ) :
            line = IndiIDs[0][i] + "\t" + IndiIDs[1][i]

            file.write( line +  "\n")  
            
            
    # write out the final r^2
    yhat_all += knet_main.EPSILON # for numerical stability
    rSQ = np.corrcoef( ytrue, yhat_all, rowvar=0)[1,0]**2      
       
    with open(outLoc_PRS, "w") as file: 
            file.write(str(rSQ) )   