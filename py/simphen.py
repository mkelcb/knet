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

import numpy as np

# Functions that help to generate phenotypes

np.var

def generatePhenotypes(h2, causalIndices, standardisedGenotypes, randomSeed =1) :
  
    np.random.seed(randomSeed)
    numCausals = len(causalIndices)
    numIndividuals = standardisedGenotypes.shape[0]
      
     # 1. Generate 'pure' effect sizes from the distribution
    effectSizes = np.random.normal(size=numCausals) # effect sizes for SNPs
    
      # 2. calculate 'pure' breeding value (g) for each individual
    g = np.zeros(numIndividuals)
    
    
    for i in range(numIndividuals): # loop through all individuals: 
        for j in range(numCausals): # loop through all their alleles which are causal, 
            # get individual 'i' 's 'J' allele count, and multiply it by that alleles' pure effect size
            g[i] = g[i]+ standardisedGenotypes[i, causalIndices[j] ] * effectSizes[j]
        	
    
      
    # 3. calculate 'g2'
    var_g = np.var(g) # calculate varaince of the breeding values
    # calculate the root of the ratio of heritability and breeding value variance
    h2_denom =np.sqrt( h2 / var_g ) # this is the 'denominator' for h2, IE 1/(Vg+Ve)
      
    # loop through breeding values and calculate g2 as: g/res
    g2 =np.zeros(numIndividuals)
    for i in range(numIndividuals):
        g2[i] = g[i] * h2_denom 
    
      
    # 4. calculate raw phenotypes
    noise = np.random.normal(loc = 0.0, scale = np.sqrt(1.0 - h2) , size = numIndividuals)
    pheno = g2 + noise  # add some gaussian noise to the breeding values to dilute them IE: Y = g + e
    
    # 5. standardise them into z-scores (mean=0, var=0)
    pheno = (pheno - np.mean(pheno)) / np.std(pheno)
      
      
    return ({"PHENO":pheno,"EFFECTS":effectSizes})



#causals = pickCausalIndices(10, 100, c(0.333,0.333,0.333), c(0.333,0.333,0.333))
#i =1
#numSNPs = 100
#numCausals = 10
#regions = c(0.25,0.5,0.25)
#regionH2= c(0.8,0.1,0.1)

#regions = np.array([0.3,0.2])
#if np.sum(regions) < 0.999 : raise ValueError("regions don't sum to 100% ")
#numSNPs = 5
#allIndices = np.array(range(0,numSNPs) )



def pickCausalIndices(numCausals, numSNPs, regions = None, regionH2 = None, randomSeed =1) :
  
    allIndices = np.array(range(0,numSNPs) ) # create an array for each locus index
    np.random.seed(randomSeed)
    print("regions is: " + str(regions))
    if regions is not None: # if we have specified regional h2
        # do some QC
        if np.sum(regions) < 0.999 : raise ValueError("regions don't sum to 100% ")
        if regions is regionH2: raise ValueError("no regional h2 specified")
        if np.sum(regionH2) < 0.999 : raise ValueError("h2 doesn't sum to 100% ")
        if len(regions) != len(regionH2) : raise ValueError("You need to specify a h2 content for each region") 
        
        # split te SNPs into regions of the desired size
        numRegions = len(regions)
        allRegionIndices = None
        
        regionEnd = 0
        numCausalsSoFar = 0
        for i in range(numRegions):
            print("picking causals for region: " + str(i))
            regionStart = int(np.round(regionEnd))  # region starts at where the previous region finished +1
            regionSize = int(np.round(regions[i] * numSNPs)) # 0.33 * 100 = 33, IE if all regions are 1/3, then we will miss 1 SNP at the ned, but that is OK
            regionEnd =int( np.round(regionEnd + regionSize)) 
            numCausalsinRegion = np.round(numCausals * regionH2[i])
              
            if i == (numRegions-1) :  # in the last region we want to make sure rounding errors are taken care of
                regionEnd =  numSNPs  # in case of rounding errors, make sure that the last region's end is the very last snp
                numCausalsinRegion = numCausals-numCausalsSoFar # make sure that the last region's number of causals, adds up to the total required
            
            numCausalsSoFar = numCausalsSoFar+ numCausalsinRegion
              
            print("for region: " +str(i)+ " / regionStart: "+str(regionStart)+ " / regionEnd: "+ str(regionEnd))
            print("numCausalsinRegion: "+ str(numCausalsinRegion) + " / regionsize: " + str(regionSize))

            regionIndices = allIndices[regionStart:regionEnd]
            if numCausalsinRegion > len(regionIndices) : raise ValueError("not enough SNPs("+str(len(regionIndices))+") in region to pick causals("+str(numCausalsinRegion)+") from ") 
            
            causalsInRegion = pickCausals(int(numCausalsinRegion),regionIndices)
 
            if(allRegionIndices is None) : allRegionIndices =causalsInRegion
            else : allRegionIndices = np.append(allRegionIndices, causalsInRegion)
        
        return(allRegionIndices)
    
    else : 
        print("there are no regions")
        return( pickCausals(numCausals, allIndices) )  # if there were no regions, we just pick the set number of causals




#numCausals = 5
#indicesInRegion = 1:10
def pickCausals(numCausals, indicesInRegion) :
    causals = np.random.choice(indicesInRegion, int(numCausals),replace=False)
    causals = np.sort(causals) # want these to be increasing
    return(causals)


