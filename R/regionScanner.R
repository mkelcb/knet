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


debugSource("emma_edit.R")


# goes through a genome, tests for if there is any h2, in each block, which might be overlapping
# at the end they are merged
findRegions = function(y, M, blockSize = 10, stride = 5,  X = NULL) { # X is the fixed effects here, M is the GWAS design matrix
  n = nrow(M)
  p = ncol(M)
  

  # do some QC
  #if( p %% blockSize != 0 ) {stop("genome length isn't divisible by block size") }
  #if( p %% stride != 0 ) {stop("genome length isn't divisible by stride") }
  if( is.null(X)) { # This algo will NOT run if there are no Fixed effects, so if there were none passed in, then we add an intercept ( a column of 1s)
    print("no fixed effects specified, adding an intercept")
    X = matrix(1, length(y), ncol= 1)
  }

  print(paste("blockSize is:", blockSize))
  # numBlocks = (p / stride)  -1 # assuming that the whole thing is divisible by 5...
  #!? INcorrect, as if we have overlapping blocks then we will have 'stride' many of them??? 
  
  endPos = 0
  startPos = 1
  blocks_withH2 = list()
  #blocks_wit_NO_hH2 = list()
  index_h2 = 1
  index_noh2 = 1
  
  Standardised_SNPs = standardise_Genotypes(M) # Z-score them ( it is good enough to standardise these just once outside of the loop)
  
  i = 1
  #for( i in 1:numBlocks) { # go through each block of the genome ( later I will have to treat different Chroms differently)
    
  while(TRUE) { # go through each block of the genome ( later I will have to treat different Chroms differently)
    # 1 pick start/end of current block
    endPos = startPos + blockSize -1
    
    # check if we have reached the end IE if endpos > last SNP
    if(endPos > p) { 
      endPos = p
      print(paste( "reached the end of the genome, last endposition is:", endPos) )
     } 
    
    print(paste("testing block ", i , " / startpos:" , startPos , " / endPos:", endPos, " / ", p))
    
    # 2 subset genome to this block
   # M_region = M[,startPos:endPos]
    #Standardised_SNPs = standardise_Genotypes(M_region) # Z-score them
    
    M_region = Standardised_SNPs[,startPos:endPos]

    
    
    # 3. create kinship matrix from block
    K <- calc_Kinship( M_region  )
    
    
    # 4. check if there is any h2 in this block via EMMA
    results = emma.REMLE_GWAS(y, K)
    REML_LL = results$REML
    p_value = significanceTest_REML(REML_LL, y, X)
    
    # 5. depending on if there was anything in them
    if(p_value < 0.05) { # if REML is better (IE there is h2 in region)
      blocks_withH2[[index_h2]] = c(startPos,endPos)
      index_h2 = index_h2 +1
    }# else { # if ML is better (IE there is NO h2 in region)
   #   blocks_wit_NO_hH2[[index_noh2]] = c(startPos,endPos)
   #   index_noh2 = index_noh2 +1
   # }
    
    # update start position for next round
    startPos = startPos+stride 
    if(endPos >= p ) { # if the new start position would be beyond the end then we stop
      break
    }
    i =i +1
  }
  
  
  # 6. merge contiguous blocks into regions 
  regions_withh2 = mergeBlocks(blocks_withH2)
  background = booleanRegions(regions_withh2,ncol(M)) # mergeBlocks(blocks_wit_NO_hH2)# background may NOT be a continuous thing
  
  
  #7. Go through the regions we have just found and calculate the overall h2 in them
  regionH2s = vector()
  regionDeltas = vector()
  for(i in 1:length(regions_withh2) ) {
    startPos = regions_withh2[[i]][1]
    endPos = regions_withh2[[i]][2]
    M_region = Standardised_SNPs[,startPos:endPos]
    
    
    
    # create kinship matrix from block
    K <- calc_Kinship( M_region  )
    
    
    # check if there is any h2 in this block via EMMA
    results = emma.REMLE_GWAS(y, K)
    regionH2s[i] = results$vg / ( results$vg + results$ve)
    regionDeltas[i] = results$delta
  }
  

  allBackgroundDeltas = vector()
  if( length(background) > 0 ) {
    print("there is at least 1 background region")
    # go through background
    BackgroundRegion = NULL
    for(i in 1:length(background) ) { # first merge all indices, to create a single Design matrix
      
      startPos = background[[i]][1]
      endPos = background[[i]][2]
      M_region = Standardised_SNPs[,startPos:endPos]
      BackgroundRegion = cbind(BackgroundRegion,M_region )
      
      # we also want a list of all the background regions separately
      K <- calc_Kinship( M_region  )
      results = emma.REMLE_GWAS(y, K)
      allBackgroundDeltas[i] = results$delta
      
    }
  
  
    # 3. look at the overall h2 across all the background regions merged together
    K <- calc_Kinship( BackgroundRegion  )
    
    
    # 4. check if there is any h2 in this block via EMMA
    results = emma.REMLE_GWAS(y, K)
    background_h2 = results$vg / ( results$vg + results$ve)
    background_delta = results$delta
    
    

    
  }
  else { # edgecase: where there is no bg
    background_h2 = 0
    background_delta = 0
    allBackgroundDeltas[1] = background_delta
  }
  # return results,     regions with h2  BG regions           h2 in each region  deltas in in each              deltas of each BG region   overall H2 in all Bgs  overall delta in all BGs
  return (list(REGIONS=regions_withh2,BACKGROUND=background, REGIONH2=regionH2s, REGIONDELTAS=regionDeltas, BG_DELTAS=allBackgroundDeltas, BG_H2=background_h2, BG_DELTA=background_delta))
  
}


# takes the h2/background regions, and 'interweaves' them so that regions follow each other in the sequence they are on the genome (also maintaining a separate array of their matching deltas)
interweaveRegions = function(regions1,regions2, region1Deltas,region2Deltas) {

  # handle edge case of having only one region
  if(length(regions1) < 1) { return(list(ALLREGIONS=regions2,ALLDELTAS=region2Deltas))}
  if(length(regions2) < 1) { return(list(ALLREGIONS=regions1,ALLDELTAS=region1Deltas))}
  
  
  # deep copy both lists (as we will be eliminating them)
  regions1_remaining = regions1 # in R everything is copied...
  regions2_remaining = regions2
  region1Deltas_remaining = region1Deltas
  region2Deltas_remaining = region2Deltas
  
  
  # these will hold the final 'interweaved' regions
  allRegions =list()
  allDeltas = vector()
  
  
  while(TRUE) { #  there is at least 1 in each list
    
    # get next one with the one that has the lowest startPos, remove this from list
    if(regions1_remaining[[1]][1] <= regions2_remaining[[1]][1]) { # if its regions1...
      # add this , and its matching delta into the 'interweaved' list (also do delta)
      allRegions[[ ( length(allRegions) +1) ]] = regions1_remaining[[1]] # add first element to end of allregions
      allDeltas[(length(allDeltas) +1)] = region1Deltas_remaining[1]
      regions1_remaining = regions1_remaining[-1] # remove first element
      region1Deltas_remaining = region1Deltas_remaining[-1] # remove first element

    } else { # if its regions2...
      allRegions[[ ( length(allRegions) +1) ]] = regions2_remaining[[1]] # add first element to end of allregions
      allDeltas[ (length(allDeltas) +1)  ] = region2Deltas_remaining[1]
      regions2_remaining = regions2_remaining[-1] # remove first element
      region2Deltas_remaining = region2Deltas_remaining[-1] # remove first element 
    }
    
    # if we have ran out of either lists
    if(length(regions1_remaining) < 1) { # if we ran out of regions1
      allRegions= c(allRegions, regions2_remaining) # then the remainder of regions will be what is left of regions1, so we concat 
      allDeltas = c(allDeltas, region2Deltas_remaining)
      break
      }
    else if(length(regions2_remaining) < 1) {  # if we ran out of regions2
      allRegions= c(allRegions, regions1_remaining) # and vica versa...
      allDeltas = c(allDeltas, region1Deltas_remaining)
      break
    }
  }
  
  return(list(ALLREGIONS=allRegions,ALLDELTAS=allDeltas))
}



booleanRegions = function(regions, totalLength) { 
  
  negativeRegions = list()
  lastContigRegion = 0

  # edge case: check if the 1st region does not start at 1
  if( regions[[1]][1] > 1 ) {
    lastContigRegion = lastContigRegion+1
    negativeRegions[[lastContigRegion]] = c(1, (regions[[1]][1] -1) ) # start region at 1, and ends -1 before the 1st region
    print("first region does not start at 1")
    }
   # add the first block ## need to check if there are at least 2 blocks.. etc
  
  # go through each region,  
  if( (length(regions)) > 1 ) { ## edge case there must be at least 2 regions
  for(i in 1: (length(regions)-1)  ) { # go from first to 1 before last
   
    # new 'negative' region starts +1 after the endPos of the current region, and ends -1 of the next region's start
    lastContigRegion = lastContigRegion+1
    negativeRegions[[lastContigRegion]] = c(regions[[i]][2] +1, (regions[[i+1]][1] -1) )
   print(paste("negative region found at Start pos:", negativeRegions[[lastContigRegion]][1] , "/ endpos:",  negativeRegions[[lastContigRegion]][2]))
     
  }
  } else {print("there is only 1 region")}
  
  # edge case: check if last region does NOT end at the end, but there is some leftover
  
  if( regions[[length(regions)]][2] < totalLength ) {
   # print("last region does not end at the end")
    lastContigRegion = lastContigRegion+1
    negativeRegions[[lastContigRegion]] = c((regions[[length(regions)]][2]+1), totalLength ) # start region +1 to the end of last, and ends at the end
  }
  
  
  return(negativeRegions)
}




mergeBlocks = function(blocks) {
  
  regions = list()
  lastContigRegion = 1
  regions[[lastContigRegion]] = blocks[[1]] # add the first block ## need to check if there are at least 2 blocks.. etc
  
  for(i in 2:length(blocks)) { # go through each block, from the 2nd onwards
    
    # check if this block is continuous with next
    nextBlock = blocks[[i]]
    if(regions[[lastContigRegion]][2] >= nextBlock[1]-1) { # if the end position of a contig is greater than the last position before next block's start, then they are continuous
      # if yes, add this to current region
      regions[[lastContigRegion]][2] =nextBlock[2] # set the last contigs's endpoint to be the next block's endpoint
     # print(paste("merging block", i ,"into last region"))
    } else { # if not create new region
      #print(paste("contig region ended, starting new one at block", i))
     # blocks[[1]]
      
      lastContigRegion =  lastContigRegion +1
      regions[[lastContigRegion]]= blocks[[i]]
    }
  }
  return(regions)
  
}



#debugSource("emma_edit.R") # the main NN framework
#uga = findRegions(y, M, blockSize = 10, stride = 5)
#regions = uga[[2]]

#visualiseH2(resultsh2$EFFECTS, indices,numSNPs )
#visualiseRegions(uga[[1]], numSNPs)
#visualiseRegions(uga[[2]], numSNPs, TRUE)


#regionsh2 = uga[[1]]
#regions_bg = uga[[2]]
#length(regionsh2)
#length(regions_bg)
