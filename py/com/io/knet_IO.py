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
import struct
import pyplink
from pathlib import Path
import collections

##################################################################################################
# PLINK Binary / Phenotype Load
##################################################################################################
#location= "C:/Users/leo/Google Drive/PHD/0Rotations/!0Implementation/tests/knet/py/data/toy/wtccc2_hg19_toy"
#results = loadPLINK(location, recodeCaseControl = False)
def loadPLINK(location, loadPhenos = True, caseControl = True, recodeCaseControl = True, replaceMissing = False) :
    # load plink file
    bed = pyplink.PyPlink(location, mode='r', bed_format='SNP-major')
    bim = bed.get_bim()
    fam = bed.get_fam()

    if loadPhenos :
        # process phenotypes
        status= fam['status'] 
        
        if caseControl : # if trait is binary
            if recodeCaseControl : y = recodeCaseControlQuantitative(status) # if we want to treat it as a quantitative trait
            else : y = recodeCaseControlOneHot(status)# otherwise use the 'softmax' format
        else : y = np.array(status)# otherwise just leave it alone
    else : y = None # if there are no phenotypes
    
    # load genotypes from PLINK into RAM
    M = None
    loci_names = list()
    # Iterating over all loci
    for loci_name, genotypes in bed:
        if replaceMissing : genotypes[genotypes==-1] = 0  # replace no calls with 0
        if(M is None) : 
            M = genotypes
        else : 
            M = np.column_stack( (M, genotypes) )
            
        loci_names.append(loci_name)
        
    return ( {"y" : y, "M" : M, "rsid" : loci_names} ) # return results


def recodeCaseControlOneHot(status) :
    # transcode phenotype into 1 hot array
    y = np.zeros( (len(status),2)  )
    for i in range(len(status)) :
        result = np.array([0,0])
        result[ (status[i] -1) ] =1 # transocde phenotype into one-of-k format: in plink case =2, control =1, so we set the element to 1 to at the inde-1 position
        y[i] =result  
    
    return(y)


def recodeCaseControlQuantitative(status) : # transcode binary phenotype into a quantitative phenotype ( IE just turn the 1/2 into 0.0 and 1.0 )
    y = np.zeros( (len(status))  )
    for i in range(len(status)) :
        y[i] =(status[i] -1)  
    
    return(y)


#location= "C:/Users/leo/Google Drive/PHD/0Rotations/!0Implementation/tests/knet/py/data/toy/wtccc2_hg19_toy.pheno"
#results = loadPLINKPheno(location, caseControl = False, recodeCaseControl = False)
def loadPLINKPheno(location, caseControl = True, recodeCaseControl = True) :
    status = list()

    with open(location, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            status.append( np.float64(itmp[2]) )
 
    if caseControl : # if trait is binary
        if recodeCaseControl : y = recodeCaseControlQuantitative(status) # if we want to treat it as a quantitative trait
        else : y = recodeCaseControlOneHot(status)# otherwise use the 'softmax' format                                      
    else : y = np.array(status)# otherwise just leave it alone
    
    return ( y )
    

# column binds Genotype matrices (IE stacks them next to each other)
#M_list = [ results2["M"], results["M"] ]
#allM = concatChroms(M_list)
def concatChroms(M_list) : 
    return(np.concatenate(M_list, axis=1))
    
##################################################################################################
# Generic data IO (IE any kind of matrix or array)
##################################################################################################

# writes a matrix onto disk in a binary (2 files, 1 that stores the dimensions, the other the binary data)
def writeMatrixToDisk(location,data, dataType ="d") :
    # get dimensions of matrix
    nrows = data.shape[0]
    ncols = data.shape[1]
    
    # write the dimensions onto disk
    with open(location + ".id", "w") as idFile: 
        idFile.write(str(nrows) + "\t" +str(ncols) )
        
    # flatten matrix
    flat = data.ravel()

    flatData = struct.pack(dataType*len(flat),*flat  )
    with open(location + ".bin", "wb") as flat_File: 
        flat_File.write(flatData) 
    

# loads matrix from disk ( that was written by the above)
def loadMatrixFromDisk(location, dataType ="d") :

    # load id file to get dimensions
    with open(location + ".id", "r") as idFile:
        itmp = idFile.readline().rstrip().split()
        nrows = int(itmp[0])
        ncols = int(itmp[1])
        
    # how many elements to expect in the binary in total
    totalNum =nrows * ncols
    
    # open binary file
    with open(location + ".bin", "rb") as BinFile:
        BinFileContent = BinFile.read()
    
    # reformat data into correct dimensions
    flat = np.array( struct.unpack(dataType*totalNum, BinFileContent  ) )
    data = flat.reshape(nrows,ncols)
    return(data)


# writes an array onto disk ( 2 files, 1 text file that stores the length, the other the binary)
def writeVectorToDisk(location,data, dataType ="d") :
    # write the dimensions onto disk
    with open(location + ".id", "w") as idFile: 
        idFile.write( str( len(data) ) )

    flatData = struct.pack(dataType*len(data),*data  )
    with open(location + ".bin", "wb") as flat_File: 
        flat_File.write(flatData) 
    
    
# loads array from disk ( that was written by the above function)
def loadVectorFromDisk(location, dataType ="d") :

    # load id file to get dimensions
    with open(location + ".id", "r") as idFile:
        itmp = idFile.readline().rstrip()
        totalNum = int(itmp) # how many elements to expect in the binary in total

    # open binary file
    with open(location + ".bin", "rb") as BinFile:
        BinFileContent = BinFile.read()
    
    # reformat data into correct dimensions
    flat = np.array( struct.unpack(dataType*totalNum, BinFileContent  ) )
    return(flat)

##################################################################################################
# GCTA formatted kinship, as these are Lower Triangle matrices they have to be treated separately
##################################################################################################


# writes a kinship matrix onto disk in the GCTA format
def writeGCTA_GRM(location,K, id_list, N_vals) :
    # generate filenames
    BinFileName = location+".grm.bin"
    IDFileName = location+".grm.id"
    NFileName = location+".grm.N.bin"
    # how many to write
    numIndis = len(id_list[0])
    numPairs = int(  numIndis*(numIndis+1)/2 )
    
    # write ID list
    with open(IDFileName, "w") as idFile: 
        for i in range( len(id_list[0]) ):
            idFile.write(id_list[0][i] + "\t" + id_list[1][i] + "\n") 
    
    # write N file
    NFileData = struct.pack("f"*numPairs,*N_vals  )
    with open(NFileName, "wb") as NFile: 
        NFile.write(NFileData) 
    
    # write GRM to disk: unravel the GRM's lower triangle
    grm_indices = np.tril_indices(numIndis)
    grm_pairs= list (K[grm_indices[0], grm_indices[1]] )
    
    # write out binary
    GRMFileData = struct.pack("f"*numPairs,*grm_pairs  )
    with open(BinFileName, "wb") as GRM_File: 
        GRM_File.write(GRMFileData) 
    

# loads a kinship matrix from disk in the GCTA format ( written out by the above function)
def loadGCTA_GRM(location) :
    # generate filenames
    BinFileName = location+".grm.bin"
    IDFileName = location+".grm.id"
    NFileName = location+".grm.N.bin"
    
    # load the familiy/individual IDs
    id_list = list()
    id_list.append([])
    id_list.append([])
    with open(IDFileName, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            id_list[0].append(itmp[0])
            id_list[1].append(itmp[1])
    
    # how many peopple in total, IE the number of entries in the binary file
    n_subj = len(id_list[0])
    nn = int(  n_subj*(n_subj+1)/2 )
    
    # load Binary
    with open(BinFileName, "rb") as BinFile:
        BinFileContent = BinFile.read()
    
    # reformat it into the correct dimensions ( as only the lower triangle was stored flattened)
    K = np.zeros((n_subj, n_subj))
    
    grm_vals = list(struct.unpack("f"*nn, BinFileContent  ))
    inds = np.tril_indices_from(K)
    K[inds] = grm_vals # reconstruct the full matrix from the LT
    K[(inds[1], inds[0])] = grm_vals
      
    # load the 'N' file
    with open(NFileName, "rb") as NFile:
        NFileContent = NFile.read()
    
    N_vals = list(struct.unpack("f"*nn, NFileContent  ))
    
    return ( {"K" : K, "ids" : id_list, "N" :N_vals} ) # return results


##################################################################################################
# write regions
##################################################################################################



def writeRegionData(location,regions, deltas) :
    fileName = location
    with open(fileName, "w") as file: 
        for i in range( len(deltas) ):
            file.write( str(regions[i][0]) + "\t" + str(regions[i][1]) + "\t" + str(deltas[i])  + "\n")   

def loadRegionData(location) :
    fileName = location
    regions = list()
    deltas = list()
    
    with open(fileName, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            regions.append( np.array( [int(itmp[0]) , int(itmp[1])] ) )
            deltas.append( np.float64(itmp[2]) )
            
    return ( {"REGIONS":regions, "DELTAS":deltas } )







##################################################################################################
# write/load eigen Summaries of kinships
##################################################################################################

    
    
def writeEigenSum(location, eigSums) :
    for i in range( len(eigSums) ):
        writeMatrixToDisk(location + "eigVec" + str(i+1) , eigSums[i].vectors)
        writeVectorToDisk(location + "eigVal" + str(i+1), eigSums[i].values)


 
def loadEigenSum(location) :
    eigSums = list()
    moreFiles = True
    counter = 1
    while moreFiles : # keep going until we run out of files to load
        currentLocation = location + "eigVec" + str(counter) + ".id"  # files are expected to be named as eigVec1.id/bin etc
        my_file = Path(currentLocation)
        if my_file.is_file(): # check if it exists
        
            filename = location + "eigVec" + str(counter)
            vectors = loadMatrixFromDisk(filename)
            filename = location + "eigVal" + str(counter)
            values = loadVectorFromDisk(filename)
            
            eigenSummary = collections.namedtuple('values', 'vectors')
            eigenSummary.values = values
            eigenSummary.vectors =vectors
 
            eigSums.append( eigenSummary ) 
        else : moreFiles = False
        counter = counter +1
        
    return(eigSums)
    
    

    
    
    
    
##################################################################################################
##################################################################################################

# location="dummyregions" # local test
#deltas = [1,2]
#regions = list()
#regions.append( np.array([0,50]))
#regions.append( np.array([25,75]))
#writeRegionData(location,regions, deltas)
# results = loadRegionData(location)

# local testing
#gcta_data2 = loadGCTA_GRM(location)
#location = "data/gcta/output/wtccc2_hg19_toy"
# writeGCTA_GRM(location, gcta_data["K"], gcta_data["ids"], gcta_data["N"])
    

    
# local test
#location = "data/testMatrix"
#data = np.random.normal(size=[ 100,100], scale=1)

#location = "data/testVec"
#dataVec = np.random.normal( size=50, scale=1)