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

import copy
import numpy as np
exec(open("./emma_edit.py").read()) 

# goes through a genome, tests for if there is any h2, in each block, which might be overlapping
# at the end they are merged
def findRegions(y, M, blockSize = 10, stride = 5,  X = None) : # X is the fixed effects here, M is the GWAS design matrix
    n = M.shape[0]
    p = M.shape[1]
  

  # do some QC
  #if( p %% blockSize != 0 ) {stop("genome length isn't divisible by block size") }
  #if( p %% stride != 0 ) {stop("genome length isn't divisible by stride") }
    if X == None:  # This algo will NOT run if there are no Fixed effects, so if there were none passed in, then we add an intercept ( a column of 1s)
        print("no fixed effects specified, adding an intercept")
        X =  np.ones( (y.shape[0], 1) ) # make sure that this is a 2D array, so that we can refer to shape[1]
        

    print("blockSize is: " + str(blockSize) )

    endPos = 0
    startPos = 0
    blocks_withH2 = list()
    #blocks_wit_NO_hH2 = list()
    index_h2 = 0

      
    Standardised_SNPs = standardise_Genotypes(M) # Z-score them ( it is good enough to standardise these just once outside of the loop)
  
    i = 0
  
    while(True) : # go through each block of the genome ( later I will have to treat different Chroms differently)
        # 1 pick start/end of current block
        endPos = startPos + blockSize # -1
            
        # check if we have reached the end IE if endpos > last SNP
        if(endPos > p) :
            endPos = p
            print("reached the end of the genome, last endposition is: " + str(endPos) )
             
            
        print("testing block "+ str(i) + " / startpos: " + str(startPos) + " / endPos: ", str(endPos), " / "+ str(p) )
        
        # 2 subset genome to this block
        M_region = Standardised_SNPs[:,startPos:endPos]


        # 3. create kinship matrix from block
        K = calc_Kinship( M_region  )
    
        # 4. check if there is any h2 in this block via EMMA
        results = REML_GWAS(y, K)
        REML_LL = results["REML"]
        p_value = significanceTest_REML(REML_LL, y, X)
        
        # 5. depending on if there was anything in them
        if(p_value < 0.05) : # if REML is better (IE there is h2 in region)
          blocks_withH2.append( np.array([startPos,endPos]) )
          index_h2 = index_h2 +1

        
        # update start position for next round
        startPos = startPos+stride 
        if(endPos >= p ) : # if the new start position would be beyond the end then we stop
          break
        
        i =i +1
 
  

    # 6. merge contiguous blocks into regions 
    regions_withh2 = mergeBlocks(blocks_withH2)
    background = booleanRegions(regions_withh2,M.shape[1]) # mergeBlocks(blocks_wit_NO_hH2)# background may NOT be a continuous thing
      
  
  #7. Go through the regions we have just found and calculate the overall h2 in them
    regionH2s = list()
    regionDeltas = list()
    
    for i in range(len(regions_withh2)):
        startPos = regions_withh2[i][0]
        endPos = regions_withh2[i][1]
        M_region = Standardised_SNPs[:,startPos:endPos]
    
        # create kinship matrix from block
        K = calc_Kinship( M_region  )
               
        # check if there is any h2 in this block via EMMA
        results = REML_GWAS(y, K)
        regionH2s.append(results["vg"] / ( results["vg"] + results["ve"]))
        regionDeltas.append(results["delta"])
        
      



    allBackgroundDeltas = list()
    if( len(background) > 0 ) :
        print("there is at least 1 background region")
        # go through background
        BackgroundRegion = None
        for i in range(len(background)): # first merge all indices, to create a single Design matrix
            startPos = background[i][0]
            endPos = background[i][1]
            M_region = Standardised_SNPs[:,startPos:endPos]
            if(BackgroundRegion is None) : BackgroundRegion =M_region
            else : BackgroundRegion = np.append(BackgroundRegion, M_region, axis=1)
              
            # we also want a list of all the background regions separately
            K = calc_Kinship( M_region  )
            results = REML_GWAS(y, K)
            allBackgroundDeltas.append(results["delta"])
          
        
        # 3. look at the overall h2 across all the background regions merged together
        K = calc_Kinship( BackgroundRegion  )
            
            
        # 4. check if there is any h2 in this block via EMMA
        results = REML_GWAS(y, K)
        background_h2 = results["vg"] / ( results["vg"] + results["ve"])
        background_delta = results["delta"]  
    
    else : # edgecase: where there is no bg
        background_h2 = 0
        background_delta = 0
        allBackgroundDeltas.append(background_delta)
    
           # return results,     regions with h2  BG regions           h2 in each region  deltas in in each              deltas of each BG region   overall H2 in all Bgs  overall delta in all BGs
    return ( {"REGIONS":regions_withh2, "BACKGROUND":background, "REGIONH2":regionH2s, "REGIONDELTAS":regionDeltas, "BG_DELTAS":allBackgroundDeltas, "BG_H2":background_h2, "BG_DELTA":background_delta } )



# takes the h2/background regions, and 'interweaves' them so that regions follow each other in the sequence they are on the genome (also maintaining a separate array of their matching deltas)
def interweaveRegions(regions1,regions2, region1Deltas,region2Deltas) :

    # handle edge case of having only one region
    if(len(regions1) < 1) : return({"ALLREGIONS":regions2, "ALLDELTAS":region2Deltas})
    if(len(regions2) < 1) : return({"{ALLREGIONS":regions1, "ALLDELTAS":region1Deltas})
      
      
    # deep copy both lists (as we will be eliminating them)
    regions1_remaining = copy.deepcopy(regions1) # in python we have to request deep copies
    regions2_remaining = copy.deepcopy(regions2)
    region1Deltas_remaining = copy.deepcopy(region1Deltas)
    region2Deltas_remaining = copy.deepcopy(region2Deltas)
      
      
    # these will hold the final 'interweaved' regions
    allRegions = list()
    allDeltas  = list()
  
    while(True) : #  there is at least 1 in each list
        # get next one with the one that has the lowest startPos, remove this from list
        if(regions1_remaining[0][0] <= regions2_remaining[0][0]) : # if its regions1...
            # add this , and its matching delta into the 'interweaved' list (also do delta)
            allRegions.append(regions1_remaining.pop(0) ) # add first element to end of allregions, and also remove first element
            allDeltas.append(region1Deltas_remaining.pop(0) ) 
            
        else : # if its regions2...
            allRegions.append(regions2_remaining.pop(0) )  # add first element to end of allregions, and also remove first element
            allDeltas.append(region2Deltas_remaining.pop(0) ) 
        
        # if we have ran out of either lists
        if(len(regions1_remaining) < 1) : # if we ran out of regions1
            allRegions= np.concatenate( ( allRegions, regions2_remaining)  ) # then the remainder of regions will be what is left of regions1, so we concat 
            allDeltas = np.concatenate( ( allDeltas, region2Deltas_remaining) )
            break
        
        elif(len(regions2_remaining) < 1) :  # if we ran out of regions2
            allRegions= np.concatenate( ( allRegions, regions1_remaining) ) # and vica versa...
            allDeltas = np.concatenate( ( allDeltas, region1Deltas_remaining) )
            break
        

    return({"ALLREGIONS":allRegions,"ALLDELTAS":allDeltas})




def booleanRegions(regions, totalLength) :
    
    negativeRegions = list()
    lastContigRegion = 0
    
    # edge case: check if the 1st region does not start at 1
    if( regions[0][0] > 0 ) :
        lastContigRegion = lastContigRegion+1
        negativeRegions.append(  np.array([0, regions[0][0]]) ) # start region at 1, and ends -1 before the 1st region
        print("first region does not start at 1")
    
    # add the first block ## need to check if there are at least 2 blocks.. etc
      
    # go through each region,  
    if( (len(regions)) > 1 ) : ## edge case there must be at least 2 regions

        for i in range( (len(regions) -1) ):  # go from first to 1 before last
            # new 'negative' region starts +1 after the endPos of the current region, and ends -1 of the next region's start
            negativeRegions.append( np.array( [regions[i][1], regions[i+1][0]]) )
            print("negative region found at Start pos: " + str(negativeRegions[lastContigRegion][0]) + " / endpos: "+  str(negativeRegions[lastContigRegion][1]) )
            lastContigRegion = lastContigRegion+1
        
    else : print("there is only 1 region")
      
    # edge case: check if last region does NOT end at the end, but there is some leftover  
    if( regions[-1][1] < totalLength ) :
        # print("last region does not end at the end")
        lastContigRegion = lastContigRegion+1
        negativeRegions.append( np.array([ regions[-1][1], totalLength]) ) # start region +1 to the end of last, and ends at the end
    
      
      
    return(negativeRegions)



def mergeBlocks(blocks) :
  
    regions = list()
    lastContigRegion = 0
    regions.append(blocks[0]) # add the first block ## need to check if there are at least 2 blocks.. etc
      
    for i in range( 1, len(blocks) ): # go through each block, from the 2nd onwards   
        # check if this block is continuous with next
        nextBlock = blocks[i]
        if(regions[lastContigRegion][1] >= nextBlock[0]) : # if the end position of a contig is greater than the last position before next block's start, then they are continuous
            # if yes, add this to current region
            regions[lastContigRegion][1] = nextBlock[1] # set the last contigs's endpoint to be the next block's endpoint
         # print("merging block "+ str(i) +" into last region")
        else : # if not create new region
            #print("contig region ended, starting new one at block "+ str(i)) 
            lastContigRegion =  lastContigRegion +1
            regions.append(blocks[i])
        
    return(regions)
  
