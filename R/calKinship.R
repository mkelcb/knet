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


standardise_Genotypes = function(X) {
  # calculate col SD and means for SNPs
  colMeans = vector(length = ncol(X)) 
  colSDs = vector(length = ncol(X))
  
  for(i in 1:ncol(X))
  {
    colMeans[i] = mean(X[,i])
    colSDs[i] = sd(X[,i])
    
    
  }
  
  
  ## Standardise SNPs: calculate Zscores
  X_zScore = matrix(nrow =nrow(X) , ncol = ncol(X))
  for(col in 1:ncol(X)) # go through all columns
  {
    for(row in 1:nrow(X)) # go through all rows
    {
      # zScore = difference from mean,                   divided by SD
      X_zScore[row,col] =   (X[row,col] - colMeans[col] ) / colSDs[col] 
    }
  }
  
  return(X_zScore)

}


calc_Kinship = function(X) {
  p = ncol(X)
  return (  (X%*%t(X)) / p )  # 1 x 3 * 3x1 = 1x1
}