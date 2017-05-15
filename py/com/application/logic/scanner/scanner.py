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

import collections
import numpy as np
from ..reml import reml
from ..reml import kinship
#from ..utils import geno_qc
import gc

from ....application.utils import geno_qc

# from com.applicatoin.utils import geno_qc


MAXDELTA = np.exp(10) # the theoretical max for delta, IE when there is no h2 in region

# goes through a genome, tests for if there is any h2, in each block, which might be overlapping
# at the end they are merged
def findRegions(y, M, irsIds= None, blockSize = 10, stride = 5,  X = None) : # X is the fixed effects here, M is the GWAS design matrix
    #n = M.shape[0]

    # do some QC
    if X == None:  # This algo will NOT run if there are no Fixed effects, so if there were none passed in, then we add an intercept ( a column of 1s)
        print("no fixed effects specified, adding an intercept")
        X =  np.ones( (y.shape[0], 1) ) # make sure that this is a 2D array, so that we can refer to shape[1]
        
    print("Genotype matrix size in MBs is: ",geno_qc.getSizeInMBs(M)," and blockSize is: " , blockSize )
    
    regions = list()
    deltas = list()
    endPos = 0
    startPos = 0
    qc_data = geno_qc.genoQC_all(M, rsIds = irsIds)
    M = qc_data["X"]
    MAFs = qc_data["MAFs"] # need to store the original MAFs as, after standardising the genotype matrix it will be impossible to calculate MAFs anymore...
    irsIds =  qc_data["rsIds"]
    p = M.shape[1] # only set this after we have potentially removed all the bad SNPs
    Standardised_SNPs = geno_qc.standardise_Genotypes(M) # Z-score them ( it is good enough to standardise these just once outside of the loop)
    print("After standardising, matrix size in MBs is: ",geno_qc.getSizeInMBs(Standardised_SNPs) )
    #eigSum = None
   # delta = None
    i = 0
    while(True) : # go through each block of the genome ( later I will have to treat different Chroms differently)
        # 1 pick start/end of current block
        endPos = startPos + blockSize # -1
            
        # check if we have reached the end IE if endpos > last SNP
        if(endPos > p) :
            endPos = p
            print("reached the end of the genome, last endposition is: " + str(endPos) )
  
        print("numSNPS in matrix: " + str(Standardised_SNPs.shape[1])  + " // testing block "+ str(i) + " / startpos: " + str(startPos) + " / endPos: ", str(endPos), " / "+ str(p), "(", str("{0:.0f}%".format(endPos /   p   * 100) ) ,")" ) ## , end='\r'


              
          
        M_region = Standardised_SNPs[:,startPos:endPos] # 2 subset genome to this block
        K = kinship.calc_Kinship( M_region  ) # 3. create kinship matrix from block                         
        try: 
            results = reml.REML_GWAS(y, K) # 4. check if there is any h2 in this block via EMMA
            delta = results["delta"]
    
        except Exception as e :
            print("Eigenvalues won't converge, but we try once more with feelin' ( MAF filter: 0.03) ")
            print("precise error message: " , str(e))
            try: 
                M_region_filtered = geno_qc.removeRareVariants_quick(M_region, MAFs[startPos:endPos], rsIds = irsIds[startPos:endPos],minMAF = 0.03)
                K = kinship.calc_Kinship( M_region_filtered  ) # 3. create kinship matrix from block  
                results = reml.REML_GWAS(y, K)
                delta = results["delta"]
            except Exception as e: 
                delta = MAXDELTA
                print("Still won't converge, so we use default values")
                print("precise error message: " , str(e))

        print("delta for block ", str(i) , " is: " , str(delta))

        regions.append( np.array([startPos,endPos]) ) # add delta to the ones collected so far     
        deltas.append( delta )  # maybe significnce test this??              
                            
                     
        # update start position for next round
        startPos = startPos+stride 
        if(endPos >= p ) : # if the new start position would be beyond the end then we stop
          break
        
        i =i +1
        
        gc.collect()
        #del K 
       # del results


    
    # return results,     regions with h2  BG regions           h2 in each region  deltas in in each              deltas of each BG region   overall H2 in all Bgs  overall delta in all BGs
    return ( {"REGIONS":regions, "DELTAS":deltas, "rsIds":irsIds } )



def concatRegions(allRegions) : 
    deltaConcat = list()
    regionsConcat = list()
    for i in range( len(allRegions) ): # go through all regions
          deltaConcat = deltaConcat + allRegions[i]["DELTAS"] # the deltas are just simply concated
          
          # for regions we need to offset all of the next region's blocks by the last added block's endpos:
          lastPos = 0
          if( i is not 0) : lastPos = regionsConcat[-1][1]
         
          for j in range( len(allRegions[i]) ): # go through all regionss' blocks  
                allRegions[i]["REGIONS"][j] = allRegions[i]["REGIONS"][j] + lastPos # offset each block
                          
          regionsConcat = regionsConcat +  allRegions[i]["REGIONS"]
          

    return ( {"REGIONS":regionsConcat, "DELTAS":deltaConcat } )


def getDummyEigSum() :
    eigenSummary = collections.namedtuple('values', 'vectors')
    eigenSummary.values = np.array([[0,0],[0,0]])
    eigenSummary.vectors =np.array([0,0,0]) # make it deliberately mismatched so that using this will fail too


# local testing
#allDeltas = list()
#delta1 = [1,2]
#delta2 = [3,4]
#allDeltas.append(delta1)
#allDeltas.append(delta2)
#allRegions = list()
#regions1 = list()
#regions1.append( np.array([0,50]))
#regions1.append( np.array([25,75]))
#regions2 = list()
#regions2.append( np.array([0,100]))
#regions2.append( np.array([50,200]))
#allRegions.append(regions1)
#allRegions.append(regions2)
#results = concatRegions(allRegions,allDeltas)