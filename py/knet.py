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


# Dependencies:
# numpy
# scipy



# https://docs.python.org/3/library/argparse.html
#https://docs.python.org/3/howto/argparse.html 
import argparse
from pathlib import Path


from com.application.logic.knet import knet_manager
from com.application.logic.scanner import scanner
from com.application.utils import plotgen

from com.io import knet_IO
import os

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

def runh2(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet H2 analyis started')  
    if args.eig is None and args.bfile is None:
        print("either an eigen decomposition or a PLINK binary is requried")

    else :
        from com.application.logic.reml import reml
        from com.application.logic.reml import kinship
        from com.application.utils import geno_qc
        # load plink binary (or eigen summary) / phenotypes
        cc = True
        if args.cc == 0 : cc = False
       
        recodecc = True
        if args.recodecc == 0 : cc = False
        
        y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
        
        if args.eig is None : # if an eigen summary wasn't supplied, IE we don't have it
            print("calculating REML from sratch, no Eigen summary was supplied")
            genotypeData = knet_IO.loadPLINK(args.bfile, loadPhenos = False, caseControl = True, recodeCaseControl = True) 
            M = genotypeData["M"]

            qc_data = geno_qc.genoQC_all(M, rsIds = genotypeData["rsid"])
            M = qc_data["X"]
            Standardised_SNPs = geno_qc.standardise_Genotypes(M)
            K = kinship.calc_Kinship( Standardised_SNPs  ) # 3. create kinship matrix from block                         
            results = reml.REML_GWAS(y, K) # 4. check if there is any h2 in this block via EMMA
        
        else :
            print("loading saved eigen sums from: " + args.eig)
            loadedEigSum = knet_IO.loadEigenSum(args.eig)[0] # load eigen decomposition
            results = reml.REML_GWAS(y, eigenSummary = loadedEigSum) # 4. check if there is any h2 in this block via EMMA


        
        eigSum = results["eigSum"] # just resave the one we have got  
        h2 = results["vg"] / ( results["vg"] + results["ve"])
        h2_SE = reml.h2_SE_approx2(y, eigSum.values)

        print("h2: " , h2 , " / h2 SE: ", h2_SE, " / delta: ", results["delta"])
        fileName = args.out + "reml.txt"
        with open(fileName, "w") as file:
            file.write("h2=" + str(h2)  + "\n")
            file.write("h2_SE=" + str(h2_SE)  + "\n")
            file.write("delta=" + str(results["delta"])  + "\n")
            file.write("ve=" + str(results["ve"])  + "\n")
            file.write("vg=" + str(results["vg"])  + "\n")
            file.write("REML_LL=" + str(results["REML"])  + "\n")
            
        # now write out the eigen summaries too ( but only if it wasn't supplied in the first place)
        if args.eig is None : knet_IO.writeEigenSum(args.out,  [eigSum ] )  # the below function expects a list


def runKnet(args) :
    set_Threads(args)
    set_nixMem(args) 
    print('Knet Neural net started')

    # load regions
    regions =  knet_IO.loadRegionData(args.regions) # load regions


    # load plink binary / phenotypes
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : cc = False
    
    
    genotypeData = knet_IO.loadPLINK(args.knet, loadPhenos = False, caseControl = True, recodeCaseControl = True) 
    M = genotypeData["M"]
    rsIds = genotypeData["rsid"]
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
 
    # if we have a validation set
    M_validation = None
    y_validation = None
    if args.validSet :
        genotypeData = knet_IO.loadPLINK(args.validSet, loadPhenos = False, caseControl = True, recodeCaseControl = True) 
        M_validation = genotypeData["M"] 
        
        if args.validPhen :
            y_validation = knet_IO.loadPLINKPheno(args.validPhen, caseControl = cc, recodeCaseControl = recodecc) 
            

    
    # check if we need to load saved state weights
    loadedWeightsData = None
    
    if args.loadWeights :

        loadedWeightsData =list()
        moreFiles = True
        counter = 0
        
        while moreFiles : # keep going until we run out of files to load
           # currentLocation = args.loadWeights + str(counter)  # files are expected to be named as regions1.txt, regions2.txt, etc
            my_file = Path(args.loadWeights + "_"+ str(counter) +"_0.bin") # check if the main weights file exists
                          
            if my_file.is_file(): # check if it exists
                loadedWeightsData.append(list())
                
                for j in range(4) :  # there are 4 filtes, 1 for each, W, W bias, momentum and Momentub bias
                   loadedWeightsData[counter].append( knet_IO.loadMatrixFromDisk(args.loadWeights + "_"+ str(counter) + "_" + str(j)) )
                    # each weight is a matrix ( even the biases), and they are coded as name_LAYER_W/Bias/Momentum/Momentum_bias (so chrom_0_0 is layer 1's Weights_W)
                counter = counter +1
            else : moreFiles = False
            
        
        
    # pass this into knet manager, along with all the conditional params
    knet_results = knet_manager.runKnet(rsIds, y,M,regions, args.epochs, args.learnRate, args.momentum, args.evalFreq, y_validation, M_validation, args.savFreq, args.predictPheno, loadedWeightsData, args.saveWeights, args.randomSeed, args.flag, args.hidl2)
     


    # write final predictions out ( if this was requested)
    yhat = knet_results["yhat"]
    if yhat is not None :    
        fileName = args.out + "yhat.txt"
        with open(fileName, "w") as file:
            file.write("Profile"  + "\n")
            for i in range(yhat.shape[0]) :
                line = str(yhat[i][0] )
                for j in range(1, len(yhat[i]) ):
                    line = line + "\t" + str(yhat[i][j] )
                    
                file.write( line +  "\n")   # file.write(  ( str(yhat[i])[2:-1] ).replace("  ", " ").replace(" ", "\t") +  "\n")

    

    
    # write epoch results out
    results_its = knet_results["results"]["results"]              
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
    rsIds = knet_results["rsIds"]
    fileName = args.out + "nn_SNPs.txt"
    with open(fileName, "w") as file: 
        for i in range( len(rsIds)  ):
            file.write(rsIds[i]  + "\n")
            
    # write final weights out
    # results["weights"]
    weights_nn = None
    if knet_results["weights"] is not None:
        weights_nn = knet_results["weights"]
        for i in range(len(weights_nn)) :
            for j in range(len(weights_nn[i])) :
                knet_IO.writeMatrixToDisk( args.saveWeights + "_" + str(i)+ "_" + str(j)  , weights_nn[i][j])
                # each weight is a matrix ( even the biases), and they are coded as name_LAYER_W/Bias/Momentum/Momentum_bias (so chrom_0_0 is layer 1's Weights_W)
   
            

  
def runScanner(args) : 
    set_nixMem(args) 
    set_Threads(args)
    print("Knet scanner started")

    # check if we want to load the eigen decomposition summaries for each region or not
   # loadedEigSum = None
  #  if args.loadEigSum is not None :
  #      print("loading saved eigen sums from: " + args.loadEigSum)
  #      loadedEigSum = knet_IO.loadEigenSum(args.loadEigSum)
    
    cc = True
    if args.cc == 0 : cc = False
   
    recodecc = True
    if args.recodecc == 0 : cc = False

    # load plink binary / phenotypes
    genotypeData = knet_IO.loadPLINK(args.scanner, loadPhenos = False, caseControl = True, recodeCaseControl = True) 
    M = genotypeData["M"]
    y = knet_IO.loadPLINKPheno(args.pheno, caseControl = cc, recodeCaseControl = recodecc) 
 
    # obtain regions
    regionResults = scanner.findRegions(y, M, irsIds = genotypeData["rsid"], blockSize = args.filterSize, stride = args.stride,  X = None)

    # check if we want to save the eigen decomposition summaries for each region or not
 #   if args.saveEigSum is not None:
  #      print("saving eigen decompositions to: " + args.saveEigSum)
 #       knet_IO.writeEigenSum(args.saveEigSum, regionResults["eigSum"] )
        
    # write regions onto disk
    knet_IO.writeRegionData(args.out,regionResults["REGIONS"], regionResults["DELTAS"])
    print("written regions to: "  + args.out)
    
    
    # write out the SNPs that were used for the analysis
    rsIds = regionResults["rsIds"]
    fileName = args.out + "_SNPs.txt"
    with open(fileName, "w") as file: 
        for i in range( len(rsIds)  ):
            file.write(rsIds[i]  + "\n")
            

    
    
def runMerge(args) :  
    set_nixMem(args) 
    set_Threads(args)
    print('Knet merging started, from: ' + args.merge + " to: "  + args.out)
    # check for other required arguments

    location = args.merge
    outLocation = args.out
    
    moreFiles = True
    counter = 1
    allRegions = list()
    while moreFiles : # keep going until we run out of files to load
        currentLocation = location + str(counter) + ".txt" # files are expected to be named as regions1.txt, regions2.txt, etc
        my_file = Path(currentLocation)

        if my_file.is_file(): # check if it exists
            allRegions.append( knet_IO.loadRegionData(currentLocation) ) # load regions
        else : moreFiles = False
        counter = counter +1
    
    # concat them into a single list
    results = scanner.concatRegions(allRegions)
    
    # write these onto disk
    knet_IO.writeRegionData(outLocation,results["REGIONS"], results["DELTAS"] )


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
subparsers.dest = 'either knet, scanner, h2 or merge' # hack to make subparser required

# create the parser for the "a" command
parser_knet = subparsers.add_parser('knet')
parser_knet.add_argument('--knet', required=True) # the location of the train set binaries
parser_knet.add_argument("--pheno", required=True)
parser_knet.set_defaults(func=runKnet)

# knet subparams
parser_knet.add_argument("--regions", required=True)  # ,required=False   
parser_knet.add_argument("--loadWeights") # from where we want to load the weights
parser_knet.add_argument("--saveWeights") # where we wnt to save weights
parser_knet.add_argument("--savFreq", default=-1, type=int) # how frequently we make backups of Weights
parser_knet.add_argument("--epochs", default=100, type=int) # how many epochs
parser_knet.add_argument("--learnRate", default=0.005, type=float) 
parser_knet.add_argument("--momentum", default=-1, type=float)   # -1 means 'disabled'
parser_knet.add_argument("--validSet") # the location for the binaries for the validation set
parser_knet.add_argument("--validPhen") # the location for the binaries for the validation set phenotypes
parser_knet.add_argument("--evalFreq", default=10, type=int) # how frequently we evaluate prediction accuracy (-1 for disabled)                     
parser_knet.add_argument("--cc")  # ,required=False  # if phenotype is case control
parser_knet.add_argument("--recodecc")  # ,required=False       # if we want to recode case control to quantitative
parser_knet.add_argument("--randomSeed", default=1, type=int)                       
parser_knet.add_argument("--flag", default=0, type=int)     
parser_knet.add_argument("--hidl2", default=0.0, type=float)               
              
# parser_knet.add_argument("--topology", required=True) # the location of the file that describes the network's topology (IE number and size of layers etc)
parser_knet.add_argument("--predictPheno", default=-1, type=int) # if network should save phenotype predictions to a location at the end, for a validation set                  
                       
                        
parser_scanner = subparsers.add_parser('scanner')
parser_scanner.add_argument('--scanner', required=True)
parser_scanner.add_argument("--pheno", required=True)
parser_scanner.set_defaults(func=runScanner)

parser_merge = subparsers.add_parser('merge')
parser_merge.add_argument('--merge', required=True)
parser_merge.set_defaults(func=runMerge)


parser_h2 = subparsers.add_parser('h2')
parser_h2.add_argument('--eig') # the location of the eigen decomposition
parser_h2.add_argument('--bfile') # the location of the plink binaries 
parser_h2.add_argument("--pheno", required=True)
parser_h2.add_argument("--cc")  # ,required=False
parser_h2.add_argument("--recodecc")  # ,required=False     
parser_h2.set_defaults(func=runh2)
 


# scanner subparams
parser_scanner.add_argument("--stride", default=25, type=int)  # ,required=False
parser_scanner.add_argument("--filterSize", default=50, type=int)  # ,required=False
parser_scanner.add_argument("--saveEigSum")  # ,required=False
parser_scanner.add_argument("--loadEigSum")  # ,required=False
parser_scanner.add_argument("--cc")  # ,required=False
parser_scanner.add_argument("--recodecc")  # ,required=False          
                   
# retreive command line arguments
args = parser.parse_args()
args.func(args)

# toy test
# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/toyregions_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy --pheno /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy.pheno --saveEigSum /nfs/users/nfs_m/mk23/test/pytest/toyeig



# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno --saveEigSum /nfs/users/nfs_m/mk23/test/pytest/f1/22eig_
#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno

#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/21 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/21 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/21.pheno


#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/15 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15.pheno

#python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/1 --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/1 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/1.pheno



# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/15_s100_ --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15 --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/15.pheno


# python knet.py --out /nfs/users/nfs_m/mk23/test/pytest/toyregions_ --threads 8 scanner --scanner /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy --pheno /nfs/users/nfs_m/mk23/data/gwas2/toy/wtccc2_hg19_toy.pheno --loadEigSum /nfs/users/nfs_m/mk23/test/pytest/toyeig


#   python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22_s100d_ --threads 2 scanner --scanner /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22 --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno

# python /nfs/users/nfs_m/mk23/software/knet/knet.py --out /nfs/users/nfs_m/mk23/test/pytest/f1/22_s100t_ --threads 2 scanner --scanner /nfs/users/nfs_m/mk23/test/pytest/f1/22_toy_long --filterSize 100 --stride 50 --pheno /lustre/scratch115/realdata/mdt0/teams/anderson/mk23/main/folds/chroms_f1/22.pheno


##################################################################################################
##################################################################################################
# Local Tests

# SCANNER
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/regions22_','scanner', '--scanner','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno', '--stride', '50', '--filterSize', '100']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
#args.func(args)

# KNET MAIN
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10' ]) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'

 # load / save weights                   
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/chr22','knet', '--knet','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno',  '--regions', '../../../0cluster/results/knettest/regions22_', '--epochs', '10', '--learnRate', '0.00005', '--momentum', '0.9', '--validSet', '../../../0cluster/data/knettest/22_toy_long_valid' ,'--validPhen', '../../../0cluster/data/knettest/22_toy_long_valid.pheno', '--evalFreq', '10' , '--loadWeights',  '../../../0cluster/results/knettest/chr22']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
  
                        
# h2 analysis:
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/h2/chr1','h2', '--bfile','../../../0cluster/data/knettest/22_toy_long_train', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'

# same as above but  loading eigsum
args = parser.parse_args(['--out', '../../../0cluster/results/knettest/h2/chr1','h2', '--eig','../../../0cluster/results/knettest/h2/chr1', '--pheno', '../../../0cluster/data/knettest/22_toy_long_train.pheno']) # , '--loadEigSum', '../../../0cluster/data/data/toy/eig/'
 
                 
# parser_h2.add_argument('--') # the location of the eigen decomposition

                        
#        
# --saveWeights
# --savFreq


                        
                        
                        
                        
