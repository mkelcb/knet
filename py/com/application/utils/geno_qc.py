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
import sys

# performs the combined QC all at once in a single loop, so it is more efficient
# can disable individual QC filters by setting it to -1
def genoQC_all(X, rsIds = None, replaceMissing = True, minObserved = 0.95, minMAF = 0.01, minVariance = 0.02) :
    indicesToRemove = list()
    
    numIndividuals = X.shape[0] 
    MAFs = np.zeros(X.shape[1] ) # create a sparse array that will hold the MAFs
    
    for i in range(0, X.shape[1]): # go through each col
        alreadyRemoved = False # only want to remove SNPs once
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        
        if alreadyRemoved is False and minObserved != -1  :       # if we do want to filter for missingness
            count = np.sum(X[:,i] == -1)
            percMissing  = count / numIndividuals
            if percMissing >= (1 - minObserved) : 
                indicesToRemove.append(i) 
                alreadyRemoved = True
                print("SNP " + snpId + "  has too many missing(" + str(percMissing)+ ")")
       
        if replaceMissing : X[:,i][  X[:,i] ==-1] = 0  # replace no calls with 0
            
        if minMAF != -1  :       # if we do want to filter for minimum MAF,
            count = np.sum(X[:,i])
            MAF  = count / numIndividuals 
            MAFs[i] = MAF # save away MAF
            if alreadyRemoved is False and MAF < minMAF  :  #  we don't check for 'already removed' here as we definitely want to compute the MAF
                indicesToRemove.append(i)
                alreadyRemoved = True
                print("SNP " + snpId + " is has too low MAF(" + str(MAF)+ "), count: " + str(count) + " / " + str(numIndividuals))

                
        if alreadyRemoved is False and minVariance != -1  :               
            SNP_var = np.var(X[:,i]) # this is the Population variance,
            # print("variance for SNP",i , " is: " , SNP_var)
            if(SNP_var < minVariance ) : 
                indicesToRemove.append(i)
                alreadyRemoved = True
                print("SNP " + snpId + " has too low variance(" + str(SNP_var) + ")" )
        
        
    print("QC removed" , len(indicesToRemove), " SNPs out of" , X.shape[1])

    rsIds_qc = None
    if rsIds is not None :
        rsIds_qc = np.delete(rsIds, indicesToRemove)
      
    MAFs_qc = np.delete(MAFs, indicesToRemove)
    X_qc = np.delete(X, indicesToRemove, axis=1)
    
    return (  {"X":X_qc, "rsIds":rsIds_qc, "MAFs":MAFs_qc, "indicesToRemove": indicesToRemove } )


def removeList(X, indicesToRemove, rsIds = None) :
    rsIds_qc = None
    if rsIds is not None :
        rsIds_qc = np.delete(rsIds, indicesToRemove)

    X_qc = np.delete(X, indicesToRemove, axis=1)
    return (  {"X":X_qc, "rsIds":rsIds_qc} )

    
def removeRareVariants_quick(X, MAFs, rsIds = None, minMAF = 0.01) :
    indicesToRemove = list()
    for i in range(0, len(MAFs)): # go through SNPs MAF
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        MAF  = MAFs[i]

        if( MAF < minMAF ) : 
            indicesToRemove.append(i)
            print("SNP " + snpId + " is has too low MAF(" + str(MAF)+ ")")

            
    print("QC removed " , len(indicesToRemove), " SNPs out of " , X.shape[1])      
    return ( np.delete(X, indicesToRemove, axis=1) )


def replaceMissing(X) : # this modifies the ORIGINAL data
    for i in range(0, X.shape[1]): # go through each col
         X[:,i][  X[:,i] ==-1] = 0  # replace no calls with 0
        
    return (X )

def removeMissing(X, rsIds = None, minObserved = 0.95) :

    indicesToRemove = list()
    for i in range(0, X.shape[1]): # go through each col
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        count = np.sum(X[:,i] == -1)
        percMissing  = count / X.shape[0]  
        ##print("number of missing for SNP",i , " is: " , count)
        if( percMissing >= (1 - minObserved) ) : 
            indicesToRemove.append(i)
            print("SNP " + snpId + "  has too many missing(" + str(percMissing)+ ")")

    
    print("QC removed " , len(indicesToRemove), " SNPs out of " , X.shape[1])
    return ( np.delete(X, indicesToRemove, axis=1) )



def removeRareVariants(X, rsIds = None, minMAF = 0.01) :
    indicesToRemove = list()
    numIndividuals = X.shape[0] 
    print("numIndividuals:" ,numIndividuals)
    for i in range(0, X.shape[1]): # go through each col
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        count = np.sum(X[:,i])
        MAF  = count / numIndividuals

        if( MAF < minMAF ) : 
            indicesToRemove.append(i)
            print("SNP " + snpId + " is has too low MAF(" + str(MAF)+ "), count: " + str(count) + " / " + str(numIndividuals))

            
    print("QC removed " , len(indicesToRemove), " SNPs out of " , X.shape[1])      
    return ( np.delete(X, indicesToRemove, axis=1) )


def removeMonomorphicVariants(X, rsIds = None) :
    colSDs = np.zeros(X.shape[1])

    indicesToRemove = list()
    for i in range(0, X.shape[1]): # go through each col
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        colSDs[i] = np.std(X[:,i]) # this is the Population variance,
        if(colSDs[i] == 0 ) : 
            indicesToRemove.append(i)
            print("SNP " + snpId + " is monomorphic")
            
    print("QC removed " , len(indicesToRemove), " SNPs out of " , X.shape[1])        
    return ( np.delete(X, indicesToRemove, axis=1) )


def removeLowVarianceSNPs(X, rsIds = None, minVariance = 0.02) :
    indicesToRemove = list()
    for i in range(0, X.shape[1]): # go through each col
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        SNP_var = np.var(X[:,i]) # this is the Population variance,
        print("variance for SNP",i , " is: " , SNP_var)
        if(SNP_var < minVariance ) : 
            indicesToRemove.append(i)
            print("SNP " +snpId + " has too low variance" + str(SNP_var) + " )" )
      
    print("QC removed " , len(indicesToRemove), " SNPs out of " , X.shape[1])  
    return ( np.delete(X, indicesToRemove, axis=1) )



def standardise_Genotypes(X, rsIds = None) :
    # calculate col SD and means for SNPs
    colMeans = np.zeros(X.shape[1])
    colSDs = np.zeros(X.shape[1])
      
    for i in range(0, X.shape[1]): # go through each col
        snpId =  str(i) 
        if rsIds is not None : snpId = rsIds[i] # if we have supplied the Rs Ids, then we want to display the name of the ones we have removed
        colMeans[i] = np.mean(X[:,i])
        colSDs[i] = np.std(X[:,i]) # this is the Population variance, IE no correction DFs lost
        if(colSDs[i] == 0 ) : raise ValueError("SNP " + snpId + " is monomorphic")


    ## Standardise SNPs: calculate Zscores
    X_zScore = np.zeros((X.shape[0], X.shape[1]))
    for col in range(0, X.shape[1]):    # go through all columns
        for row in range(0, X.shape[0]):  # go through all rows
            # zScore = difference from mean,                   divided by SD
            X_zScore[row,col] =   (X[row,col] - colMeans[col] ) / colSDs[col] 
        
   
      
    return(X_zScore)
    


# standardises this in place, if it doesn't need to convert datatypes
def standardise_Genotypes_01(X, convertDataType = -1) :
    print("standardise Genotypes to be within 0-1, rather than Z-scores")
    
    if convertDataType != -1 : 
        print("converting datatype to " , convertDataType)
        X = X.astype(convertDataType)
    
    for i in range(0, X.shape[1]): # go through each col
        x = X[:,i]
        max_x = np.max(x)
        min_x = np.min(x)
        denominator = (max_x  - min_x)
        if denominator != 0 : X[:,i] = (x -min_x )/ denominator # watch out fo division by 0
        

    return(X)


def getSizeInMBs(myObject) :
    return ( np.round( sys.getsizeof(myObject) / 1024/ 1024 )  )

def getSizeInGBs(myObject) :
    return ( np.round( sys.getsizeof(myObject) * 10 / 1024/ 1024 / 1024 ) / 10  )
# z-sclae data 
# from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train_s = sc.fit_transform(xorInput)
#X_test_s = sc.transform(X_test)