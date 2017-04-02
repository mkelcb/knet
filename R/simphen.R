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

# Functions that help to generate phenotypes

generatePhenotypes = function(h2, causalIndices, standardisedGenotypes, randomSeed =1) {
  
  set.seed(randomSeed)
  numCausals = length(causalIndices)
  numIndividuals = nrow(standardisedGenotypes)
  
 # 1. Generate 'pure' effect sizes from the distribution
  effectSizes <- rnorm(numCausals) # effect sizes for SNPs
  
  # 2. calculate 'pure' breeding value (g) for each individual
  g = vector(length = numIndividuals)
  for( i in 1:numIndividuals) { # loop through all individuals: 
 
      for(j in 1: numCausals) {# loop through all their alleles which are causal,
      
        # get individual 'i' 's 'J' allele count, and multiply it by that alleles' pure effect size
        g[i] = g[i]+ standardisedGenotypes[i, causalIndices[j] ] * effectSizes[j];
      }	
  }
  
  # 3. calculate 'g2'
  var_g = var(g) # calculate varaince of the breeding values
  # calculate the root of the ratio of heritability and breeding value variance
  scaled =sqrt( h2 / var_g )
  
 # loop through breeding values and calculate g2 as: g/res
  g2 =vector (length = numIndividuals)
  for(i in 1:numIndividuals) { g2[i] = g[i] * scaled }

  
  # 4. calculate raw phenotypes
  noise = rnorm(numIndividuals, mean = 0.0, sd = sqrt(1.0 - h2))
  pheno = g2 + noise

  # 5. standardise them into z-scores (mean=0, var=0)
  pheno = scale(pheno)
  
  
  return (list(PHENO=pheno,EFFECTS=effectSizes))
}

#causalIndices = indices
#effects = rnorm(200)
#length(effects)
#length(causalIndices)

#causalIndices = 201:400
# visualises the genomic architecture of h2
visualiseH2 = function( effects, causalIndices, numSNPs) {
  
  values = rep(0,numSNPs)
  values[causalIndices] = abs(effects)
  barplot(values, col ="green", axes = TRUE,  space =0, border = NA, xlab="SNPs", ylab="Effect size", main="h2 architecture")
  axis(1)
}

#regions = list()
#regions[[1]] = c(200,400)
#regions[[2]] = c(700,800)

# draws over a heritability plot, red bounding boxes, where it has found h2 (shouldbe called after visualiseH2)
visualiseRegions = function(  regions, numSNPs, backgroundRegions = FALSE) {
  values = rep(0,numSNPs)

  regionIndices = vector()
  for(i in 1:length(regions)) { # create all indices within a contiguous region
    regionStart = regions[[i]][1]
    regionEnd = regions[[i]][2]
    
    regionIndices = c(regionIndices, (regionStart:regionEnd) )
  }

  regionColour = "red"
  if (backgroundRegions == TRUE) {

    regionColour = "blue"
    }
  
 
  
  values[regionIndices] = 1 # maxEffect  # we want bounding boxes
  par(new=TRUE)
  barplot(values, col=makeTransparent(regionColour, alpha=0.5), axes = FALSE,  space =0, border = NA)
 
}


makeTransparent = function(..., alpha=0.5) {
  
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")
  
  alpha = floor(255*alpha)  
  newColor = col2rgb(col=unlist(list(...)), alpha=FALSE)
  
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  
  return(newColor)
  
}




#causals = pickCausalIndices(10, 100, c(0.333,0.333,0.333), c(0.333,0.333,0.333))
#i =1
#numSNPs = 100
#numCausals = 10
#regions = c(0.25,0.5,0.25)
#regionH2= c(0.8,0.1,0.1)


pickCausalIndices = function(numCausals, numSNPs, regions = NULL, regionH2 = NULL, randomSeed =1) {
  
  allIndices = 1:numSNPs # create an array for each locus index
  set.seed(randomSeed)
  
  if(!is.null(regions) ){ # if we have specified regional h2
    # do some QC
    if(sum(regions) < 0.999 ) {stop("regions don't sum to 100% ") }
    if(is.null(regionH2)) {stop("no regional h2 specified") }
    if(sum(regionH2) < 0.999 ) {stop("h2 doesn't sum to 100% ") }
    
    if(length(regions) != length(regionH2) ) {stop("You need to specify a h2 content for each region") }
    # split te SNPs into regions of the desired size
    numRegions = length(regions)
    allRegionIndices = vector()
    
    regionEnd = 0
    numCausalsSoFar = 0
    for(i in 1:numRegions) {
      regionStart = round(regionEnd+1) # region starts at where the previous region finished +1
      regionSize = round(regions[i] * numSNPs) # 0.33 * 100 = 33, IE if all regions are 1/3, then we will miss 1 SNP at the ned, but that is OK
      regionEnd = round(regionEnd + regionSize)
      numCausalsinRegion = round(numCausals * regionH2[i])
      
      if(i == numRegions) {  # in the last region we want to make sure rounding errors are taken care of
        regionEnd =  numSNPs  # in case of rounding errors, make sure that the last region's end is the very last snp
       numCausalsinRegion = numCausals-numCausalsSoFar # make sure that the last region's number of causals, adds up to the total required
        }
      numCausalsSoFar = numCausalsSoFar+ numCausalsinRegion
      
      #print(paste("for region:",i,"regionStart:",regionStart, "/ regionEnd:", regionEnd))
      #print(paste("numCausalsinRegion:",numCausalsinRegion))
      
      regionIndices = allIndices[regionStart:regionEnd]
      if(numCausalsinRegion > length(regionIndices)) { stop("not enough SNPs in region to pick causals from ")}
        
      causalsInRegion = pickCausals(numCausalsinRegion,regionIndices)
      allRegionIndices = c(allRegionIndices, causalsInRegion)
    }
    return(allRegionIndices)
    
  } else { return( pickCausals(numCausals, allIndices) ) } # if there were no regions, we just pick the set number of causals
}




#numCausals = 5
#indicesInRegion = 1:10
pickCausals = function(numCausals, indicesInRegion) {
  causals = sample(indicesInRegion, numCausals)
  causals = sort(causals) # want these to be increasing
  return(causals)
}


