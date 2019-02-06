# -*- coding: utf-8 -*-

#MIT License

#Copyright (c) 2019 Marton Kelemen

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


# Dependencies:
# numpy
# scipy



# https://docs.python.org/3/library/argparse.html
#https://docs.python.org/3/howto/argparse.html 
import argparse



#from com.application.logic.knet import knet_manager
from com.application.logic.knet import knet_manager_ols
#from com.application.logic.knet import knet_manager_keras

import os
import gc


def set_Threads(args) : 
    if args.threads is not None  :
        os.environ['MKL_NUM_THREADS'] = args.threads # '16'  # use h ere the N , where N: the number of cores acailable or limit to 1
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        os.environ['OMP_NUM_THREADS'] = '1'
        
        print("set MKL number of threads to: " + str(args.threads))
   
    
def set_nixMem(args) : 
    if args.nixMem is not None  :
        import resource # this only exists on Unix/Linux based systems
        rsrc = resource.RLIMIT_AS
        soft, hard = resource.getrlimit(rsrc)
        print('Soft limit starts as  :', soft)
        print('Hard limit starts as  :', hard)
        
        resource.setrlimit(rsrc, (args.nixMem * 1048576, hard)) #limit
        
        soft, hard = resource.getrlimit(rsrc)
        print('Soft limit changed to :', soft)
        print('Hard limit changed to  :', hard)


def runKnet(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet OLS started')
    knet_manager_ols.runKnet(args)  
    # the below will cause import errors
    #if int(args.keras) == 1 : knet_manager_keras.runKnet(args)  
    #elif int(args.pytorch) == 1 : knet_manager_pytorch.runKnet(args)  
    #else : knet_manager.runKnet(args)
    gc.collect() 


##################################################################################################
# setup COmmand line parser
##################################################################################################

parser = argparse.ArgumentParser()

# overall
parser.add_argument("--out",required=True, help='an output location is always required')
parser.add_argument("--threads",required=False, help='set number of threads used by multithreaded operations')
parser.add_argument("--nixMem",required=False, type=int, help='Memory limit for *nix based systems in Megabytes')

subparsers = parser.add_subparsers()
subparsers.required = True
subparsers.dest = 'either knet, scanner, h2, kinship, kinmerge or merge' # hack to make subparser required

# create the parser for the "a" command
parser_knet = subparsers.add_parser('knet')
parser_knet.add_argument('--knet', required=True) # the location of the train set binaries
parser_knet.add_argument("--pheno", required=True)
parser_knet.set_defaults(func=runKnet)

# knet subparams
parser_knet.add_argument("--loadWeights") # from where we want to load the weights
parser_knet.add_argument("--saveWeights") # where we wnt to save weights
parser_knet.add_argument("--savFreq", default=-1, type=int) # how frequently we make backups of Weights
parser_knet.add_argument("--epochs", default=100, type=int) # how many epochs
parser_knet.add_argument("--learnRate", default=0.005, type=float) 
parser_knet.add_argument("--momentum", default=-1, type=float)   # -1 means 'disabled'
parser_knet.add_argument("--validSet") # the location for the binaries for the validation set
parser_knet.add_argument("--validPhen") # the location for the binaries for the validation set phenotypes
parser_knet.add_argument("--evalFreq", default=10, type=int) # how frequently we evaluate prediction accuracy (-1 for disabled)                     
parser_knet.add_argument("--cc", type=int)  # ,required=False  # if phenotype is case control
parser_knet.add_argument("--recodecc", type=int)  # ,required=False       # if we want to recode case control to quantitative
parser_knet.add_argument("--randomSeed", default=1, type=int)                       
parser_knet.add_argument("--hidCount", default=0, type=int)     # number of hidden layers
  
parser_knet.add_argument("--hidAct", default=0, type=int)        # the hidden layer activations ( 1 = sigmoid, 2 = RELU, 3 = linear, 4 = softplus, 5 =  LeakyReLU, 6 =SELU)
         



parser_knet.add_argument("--batch_size", default=0, type=int)        # the size of the minibatches, use 0 for no minibatches (IE train all at once)
parser_knet.add_argument("--bnorm", default=1, type=int)        # if (spatial) batch normalization is enabled (1) or not (0)
parser_knet.add_argument("--lr_decay", default=0, type=int)        # learning rate decay should be enabled (1) or not (0)
parser_knet.add_argument("--optimizer", default=0, type=int)        # the optimizer, 0 for SGD (the default), 1 for ADAM, and 2 for AMSGrad
parser_knet.add_argument("--float", default=64, type=int)        # the float precision, valid options are 16, 32 and 64 (the default)
parser_knet.add_argument("--inference", default=0, type=int)        # if an Inference (ie deep dreaming) run is to be performed (1) or training run should be performed (0)
parser_knet.add_argument("--orig", default=0, type=int)
parser_knet.add_argument("--firstLayerSize", default=1000, type=int) # the number of units in the first layer
parser_knet.add_argument("--dropout", default=-1, type=float)  # % of units to switch off at each iteration


parser_knet.add_argument("--mns") 
parser_knet.add_argument("--sstd") 
parser_knet.add_argument("--snpIDs") 

parser_knet.add_argument("--convLayers", default=0, type=int) # how many convolutional layers to add (0 for disabled)
parser_knet.add_argument("--convFilters", default=500, type=int) # the number of filters that we will use in the first layer, each subsequent layer will have i * this many filters 
parser_knet.add_argument("--widthReductionRate", default=1, type=int) # The rate at which the network 'thins' IE if we start at 1000 neurons in layer 1, then at rate of 1 (default), we half it every layer, with a rate of 2, it will half every second layer Ie we will get two layers with 1000 units each, and then two 500 units etc

parser_knet.add_argument("--keras",required=False, help='to run keras/tensorflow backend (1) instead of original KNeT', default=0, type=int)    
parser_knet.add_argument("--pytorch",required=False, help='to run pytorch backend (1) instead of original KNeT', default=0, type=int)  
parser_knet.add_argument("--gpu",required=False, help='...', default=0, type=int)                
                        
# parser_knet.add_argument("--topology", required=True) # the location of the file that describes the network's topology (IE number and size of layers etc)
parser_knet.add_argument("--predictPheno", default=-1, type=int) # if network should save phenotype predictions to a location at the end, for a validation set                              
parser_knet.add_argument("--num_CPU", default=1, type=int) # the number of CPU cores Keras should use                  
parser_knet.add_argument("--qc", default=1, type=int) # if SNP QC is to be performed   

parser_knet.add_argument("--decov", default=0.0, type=float) # if decov regularizer should be added  to the penultimate layer (> 0) or not (0)
parser_knet.add_argument("--hidl2", default=0.0, type=float)        # the L2 regularizer shrinkage param     
parser_knet.add_argument("--l1", default=0.0, type=float) # if l1 regularizer should be added  to the penultimate layer (> 0) or not (0)
parser_knet.add_argument("--ortho", default=0.0, type=float) # if ortho v1 regularizer should be added  to the penultimate layer (> 0) or not (0)
parser_knet.add_argument("--orthov2", default=0.0, type=float) # if ortho v2 regularizer should be added  to the penultimate layer (> 0) or not (0)
      
parser_knet.add_argument("--inf_neurons", default=-1, type=int)        # if an inference layer should be added with a given size or not (-1)
 
parser_knet.add_argument("--hyperopt", default=0, type=int) # if best parameter settings are to be found via hyperopt semi-random search 


parser_knet.add_argument("--delta", default=-1.0, type=float) # if  if the ridge blup regularizer is enabled ( > -1) or not (=-1)
parser_knet.add_argument("--indices") # location of the file that lists the 'indicesToKeep' list
parser_knet.add_argument("--betas") # location of the Betas file produced by any analysis run

# retreive command line arguments
args = parser.parse_args()
args.func(args)

# toy test
# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/toyregions_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy --pheno /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy.pheno --saveEigSum /nfs/users/nfs_m/mk23/test/pytest/toyeig


# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22_s100t_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/test/pytest/f1/22_toy_long --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno


##################################################################################################
##################################################################################################
# Local Tests
#

## KNET MAIN
#args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10' ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
#
## Knet main as case control one hot                  
#args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '0.2' ,  '--hidAct'    , '2'    ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
#          
#        
# --saveWeights
# --savFreq

# LOCAL TEST

#from com.io import knet_IO
#from com.application.utils import geno_qc
#from com.application.utils import plotgen
#args = parser.parse_args(['--out', 'C:/0Datasets/knet_tempoutput/' ,'knet', '--knet', '../data/genetic/short', '--pheno', '../data/genetic/phen.pheno',  '--epochs', '21', '--learnRate', '0.0001', '--momentum', '0.9', '--evalFreq', '1',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0', '--saveWeights', 'C:/0Datasets/knet_tempoutput/w/', '--evalFreq', '1' , '--convFilters', '200', '--convLayers', '0', '--firstLayerSize' , '500' ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
#
#

       
    

#args = parser.parse_args(['--out', '../data/results/' ,'knet', '--knet', '../data/genetic/short', '--pheno', '../data/genetic/phen.pheno',  '--epochs', '21', '--learnRate', '0.0001', '--momentum', '0.9', '--evalFreq', '10',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0'   ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'



#                   
#

#
#
#
#
#

#


# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv/ --threads 10 knet --knet /root/mk23/data/genetic/short  --epochs 21 --saveWeights /root/mk23/results/knet_conv/weights/ --learnRate 0.00005 --momentum .9 --pheno /root/mk23/data/genetic/phen.pheno --recodecc 0 --cc 0 --hidAct 2 > /root/mk23/results/knet_conv.txt


# GPU run
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_new/ knet --knet /root/mk23/data/genetic/short --gpu 1 --epochs 1 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_new/weights/ --learnRate 0.00005 --momentum .9 --pheno /root/mk23/data/genetic/phenew.pheno.phen --recodecc 0 --cc 0 --hidAct 2 --hidl2 1.0 > /root/mk23/results/knet_conv_new.txt

# Inference on real data
#args = parser.parse_args(['--out', 'C:/0Datasets/NNs/genetic/inference/' ,'knet', '--knet', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/short', '--pheno', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/phenew.pheno.phen',  '--epochs', '21', '--learnRate', '0.00005', '--momentum', '0.9', '--evalFreq', '1',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0' ,'--inference', '1' ,'--loadWeights', 'C:/0Datasets/NNs/genetic/weights/' ,'--snpIndices', 'C:/0Datasets/NNs/genetic/nn_SNPs_indices.txt' ,'--mns', 'C:/0Datasets/NNs/genetic/data_mns','--sstd', 'C:/0Datasets/NNs/genetic/data_sstd','--snpIDs', 'C:/0Datasets/NNs/genetic/nn_SNPs.txt'  ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'



# GPU run on simulated data
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_sim/ knet --knet /root/mk23/data/genetic/simgeno_out --gpu 1 --epochs 81 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_sim/weights/ --learnRate 0.00005 --momentum .9 --pheno /root/mk23/data/genetic/simphe.phen --recodecc 0 --cc 0 --hidAct 2 --hidl2 1.0 > /root/mk23/results/knet_conv_sim.txt

# Inference on sim data
# args = parser.parse_args(['--out', 'C:/0Datasets/NNs/genetic/inference/' ,'knet', '--knet', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/simgeno_out', '--pheno', 'C:/Users/mk23/GoogleDrive_phd/PHD/Project/Implementation/data/genetic/simphe.phen',  '--epochs', '21', '--learnRate', '0.00005', '--momentum', '0.9', '--evalFreq', '1',  '--recodecc'    , '0' ,  '--hidCount'    , '5' ,  '--hidl2'    , '1.0' ,  '--hidAct'    , '2' , '--cc', '0' ,'--inference', '1' ,'--loadWeights', 'C:/0Datasets/NNs/genetic/weights/' ,'--snpIndices', 'C:/0Datasets/NNs/genetic/nn_SNPs_indices.txt' ,'--mns', 'C:/0Datasets/NNs/genetic/data_mns','--sstd', 'C:/0Datasets/NNs/genetic/data_sstd','--snpIDs', 'C:/0Datasets/NNs/genetic/nn_SNPs.txt'  ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'




# GPU run on simulated data with simulated validation data
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_sim_val/ knet --knet /root/mk23/data/genetic/train_data --gpu 1 --epochs 501 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_sim_val/weights/ --learnRate 0.00001 --momentum .9 --pheno /root/mk23/data/genetic/simphe_2x_train.phen --validSet /root/mk23/data/genetic/test_data --validPhen /root/mk23/data/genetic/simphe_2x_test.phen --recodecc 0 --cc 0 --hidAct 2 --hidl2 1.0 --randomSeed 42 > /root/mk23/results/knet_conv_sim_val.txt


# GPU run on simulated data with simulated validation data, that has only 5200 SNPs
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_sim_val_5200/ knet --knet /root/mk23/data/genetic/train_data_5200 --gpu 1 --epochs 101 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_sim_val_5200/weights/ --learnRate 0.001 --momentum .9 --pheno /root/mk23/data/genetic/simphe_2x_train_5200.phen --validSet /root/mk23/data/genetic/test_data_5200 --validPhen /root/mk23/data/genetic/simphe_2x_test_5200.phen --recodecc 0 --cc 0 --hidAct 2 --hidl2 1.00 --randomSeed 42 > /root/mk23/results/knet_conv_sim_val_5200.txt


# GPU run on simulated data with simulated validation data, that has only 5200 SNPs, where 50% is causal, IE massively polygenic
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_sim_val_5200_MP/ knet --knet /root/mk23/data/genetic/train_data_5200 --gpu 1 --epochs 101 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_sim_val_5200_MP/weights/ --learnRate 0.0005 --momentum .9 --pheno /root/mk23/data/genetic/simphe_2x_train_5200_MP.phen --validSet /root/mk23/data/genetic/test_data_5200 --validPhen /root/mk23/data/genetic/simphe_2x_test_5200_MP.phen --recodecc 0 --cc 0 --hidAct 2 --hidl2 1.0 --randomSeed 42 > /root/mk23/results/knet_conv_sim_val_5200_MP.txt

# This produces a 7% r^2
# python /root/mk23/model/knet/knet3.py --out /root/mk23/results/knet_conv_sim_val_5200_MP/ knet --knet /root/mk23/data/genetic/train_data_5200 --gpu 1 --epochs 101 --evalFreq 1 --saveWeights /root/mk23/results/knet_conv_sim_val_5200_MP/weights/ --learnRate 0.0005 --momentum .9 --pheno /root/mk23/data/genetic/simphe_2x_train_5200_MP.phen --validSet /root/mk23/data/genetic/test_data_5200 --validPhen /root/mk23/data/genetic/simphe_2x_test_5200_MP.phen --recodecc 0 --cc 0 --hidAct 3 --hidl2 0.5 --randomSeed 42 > /root/mk23/results/knet_conv_sim_val_5200_MP.txt



#
#from com.application.logic.knet import knet_main
#from com.application.utils import plotgen
#from com.application.utils import geno_qc
#from com.io import knet_IO
#
#
#import importlib
#from types import ModuleType
## recursively reload modules / submodules up until a certain depth ( if 2 deep it will crash or try to reload static/built in modules)
#def rreload(module, maxDepth = 2, depth = 0):
#    importlib.reload(module)
#    depth = depth +1
#    if(depth < maxDepth) :
#        for attribute_name in dir(module):
#            attribute = getattr(module, attribute_name)
#            if type(attribute) is ModuleType:
#                rreload(attribute, maxDepth, depth)
#
#
#rreload(knet_IO)
#rreload(geno_qc)
#rreload(knet_main)
#
#
##
#import importlib
#importlib.reload(geno_qc)
#

# sometimes reloading modules will fail... only solution is to restart kernel:
# http://justus.science/blog/2015/04/19/sys.modules-is-dangerous.html