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
#import pyplink
# from pyplink import *

#from ..pyplink import PyPlink
#from ..io.pyplink import PyPlink

#from ..io.pyplink import *

from ..io import pyplink

from pathlib import Path
import collections
import gc

##################################################################################################
# PLINK Binary / Phenotype Load
##################################################################################################
#location= "C:/Users/leo/Google Drive/PHD/0Rotations/!0Implementation/tests/knet/py/data/toy/wtccc2_hg19_toy"
#results = loadPLINK(location, recodeCaseControl = False)
#location = args.bfile
#males = fam.gender == 1

#for genotypes in bed.iter_geno_marker(bim.pos):
#    male_genotypes = genotypes[males.values]

#male_genotypes = genotypes[males.values]
#location = args.out
def writePLINK(location, M) :
    # load plink file
    bed = pyplink.PyPlink(location, mode='w', bed_format='SNP-major')
    for i in range(M.shape[1]):
        #print("writing SNP", i, "to file")
        bed.write_genotypes(M[:,i])
    

#male_genotypes = genotypes[males.values]
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
    M = np.zeros( (bed.get_nb_samples(),bed.get_nb_markers()) , dtype =np.int8) # pre allocate a matrix with the correct size
    #loci_names = list()
    loci_names = np.empty(bed.get_nb_markers( ), dtype =   np.dtype((str, 25)) ) ## assume no rsid will be longer than 25 chars
    # Iterating over all loci
    counter = 0
    for loci_name, genotypes in bed:
        if replaceMissing : genotypes[genotypes==-1] = 0  # replace no calls with 0
        #if(M is None) : 
        #    M = genotypes
        #else : 
        #    M = np.column_stack( (M, genotypes) )
        M[:,counter] = genotypes # much faster to just paste this in (column stack is 10x slower)
        #loci_names.append(loci_name)
        loci_names[counter] = loci_name
        counter = counter +1
       
    # produce GCTA compatible ID list
    id_list = list()
    id_list.append( list(fam["fid"]))
    id_list.append( list(fam["iid"]))  
        
    return ( {"y" : y, "M" : M, "rsid" : loci_names.tolist(), "IDs" : id_list} ) # return results


def recodeCaseControlOneHot(status) :
    # transcode phenotype into 1 hot array
    y = np.zeros( (len(status),2)  )
    for i in range(len(status)) :
        result = np.array([0,0])
        result[ int(status[i] -1) ] =1 # transocde phenotype into one-of-k format: in plink case =2, control =1, so we set the element to 1 to at the inde-1 position
        y[i] =result  
    
    return(y)


def recodeCaseControlQuantitative(status) : # transcode binary phenotype into a quantitative phenotype ( IE just turn the 1/2 into 0.0 and 1.0 )
    y = np.zeros( (len(status))  )
    for i in range(len(status)) :
        y[i] =(status[i] -1)  
    
    return(y)



def recodeOneHotCaseControl(y) :
    # transcode 1 hot array into binary
    status = np.zeros( ( (len(y), 1) ) , dtype = int )
    for i in range(y.shape[0]) :
        
        status[ i ] =  int( np.argmax(y[i,:]) + 1 )
     
    
    return(status)


#location= "C:/!0datasets/adni/wgs/glmnettest/igap05_filtered_f1_train.pheno"
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
def writeGCTA_GRM(location,K, id_list, numSNPs) :
    # generate filenames
    BinFileName = location+".grm.bin"
    IDFileName = location+".grm.id"
    NFileName = location+".grm.N.bin"
    
    # how many to write
    numIndis = len(id_list[0])
    numPairs = int(  numIndis*(numIndis+1)/2 )
    N_vals = [numSNPs] * numPairs # the number of surviving SNPs used, the number of Individuals 
    
    # write N file
    NFileData = struct.pack("f"*numPairs,*N_vals  )
    del N_vals; gc.collect();
    with open(NFileName, "wb") as NFile: 
        NFile.write(NFileData) 
    del NFileData; gc.collect();
             
                                         
    # write ID list
    with open(IDFileName, "w") as idFile: 
        for i in range( len(id_list[0]) ):
            idFile.write(id_list[0][i] + "\t" + id_list[1][i] + "\n") 
    gc.collect();

    
    # write GRM to disk: unravel the GRM's lower triangle
    grm_indices = np.tril_indices(numIndis)
    grm_pairs= list (K[grm_indices[0], grm_indices[1]] )
    del grm_indices; gc.collect();
    
    # write out binary
    GRMFileData = struct.pack("f"*numPairs,*grm_pairs  )
    del grm_pairs; gc.collect();
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
    id.close()
    
    # how many peopple in total, IE the number of entries in the binary file
    n_subj = len(id_list[0])
    nn = int(  n_subj*(n_subj+1)/2 )
    
    # load Binary
    with open(BinFileName, "rb") as BinFile:
        BinFileContent = BinFile.read()
    BinFile.close()
    
    # reformat it into the correct dimensions ( as only the lower triangle was stored flattened)
    K = np.zeros((n_subj, n_subj))
    
    grm_vals = list(struct.unpack("f"*nn, BinFileContent  ))
    inds = np.tril_indices_from(K)
    K[inds] = grm_vals # reconstruct the full matrix from the LT
    K[(inds[1], inds[0])] = grm_vals
      
    # load the 'N' file
    with open(NFileName, "rb") as NFile:
        NFileContent = NFile.read()
    NFile.close()
    
    N_vals = list(struct.unpack("f"*nn, NFileContent  ))
    
    gc.collect();
    return ( {"K" : K, "ids" : id_list, "N" :N_vals} ) # return results


def load_N_fromGCTA_GRM(location) :

    NFileName = location+".grm.N.bin"

    # load the 'N' file
    with open(NFileName, "rb") as NFile:
        NFileContent = NFile.read(4) # only load the first 4 bytes = 16 bit float, as unlike GCTA I use the same number of SNPs for all
        
    N = struct.unpack("f", NFileContent  )[0]
   
    return ( N ) 


# writes human readable text file of a Variance Compnent Analysis results
def writeVCResults(location,allModels ) :

    # write ID list
    with open(location, "w") as targetFile: 
        
        for i in range( len(allModels) ):
            line = "Model with "+ str( len(allModels[i]["vc"]) ) + " genetic variance component(s):"
            print(line) # want to save output to console too..
            targetFile.write(line + "\n")
            
            line = "BIC: " + str( allModels[i]["bic"] )
            print(line)
            targetFile.write(line + "\n") 
            
            line = "h2: " + str( allModels[i]["h2"] )
            print(line)
            targetFile.write(line + "\n") 
            
            line = "h2_CI95 lb: " +  str( allModels[i]["h2_lb"] ) + " / ub: " +  str( allModels[i]["h2_ub"] )
            print(line)
            targetFile.write(line + "\n")
            
            line = "Ve: " +  str( allModels[i]["ve"] )
            print(line)
            targetFile.write(line + "\n")
            
            epistasisCounter = 2
            for j in range( len(allModels[i]["vc"]) ) : # go through all variance components
                note = ""
                if j == 0 : note ="(additive)" # first element is always the additive VC
                elif j == allModels[i]["domcol"] : note ="(dominance)" # if j is the same col as the dominance effect
                else : 
                    note ="(epistasis "+str(epistasisCounter)+"-way)"
                    epistasisCounter = epistasisCounter +1
                    
                line = "VC"+str(j+1)+" "+note+": "+ str( allModels[i]["vc"][j] ) + " / p-value: " +  str( allModels[i]["p"][j] ) + " / vc_se: " +  str( allModels[i]["vc_se"][j] )
                print(line + "\n")
                targetFile.write(line + "\n")

            line = "_________________________________" 
            print(line)
            targetFile.write(line + "\n") 

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
# Summary Stats
##################################################################################################

#location ="C:/!0datasets/adni/wgs/stage1_p05_short_ordered.txt"
def loadSummaryStats(location) :
    fileName = location
    summaryData = list()

    with open(fileName, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            summaryData.append(  np.float64(itmp[2]) ) # the Betas are stored in the 3rd column
 
    # process them into priors
    # take their abs ( as it doesnt matter if its negative or positive)
    summaryData = np.abs(summaryData)
    
    # scale them to be 0-1
    max_x = np.max(summaryData)
    min_x = np.min(summaryData)
    denominator = (max_x  - min_x)
    if denominator != 0 : summaryData = (summaryData -min_x )/ denominator # watch out fo division by 0
   
    return ( summaryData )

#import numpy as np
#location ="C:/!0datasets/adni/wgs/h2/stage1score05_filtered_001_ordered_pruned.txt"
#priors = summaryData
def loadSummaryStats_noscale(location) :
    fileName = location
    summaryData = list()

    with open(fileName, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            summaryData.append(  np.float64(itmp[2]) ) # the Betas are stored in the 3rd column

    return ( summaryData )


##################################################################################################
# LDAK helper
##################################################################################################

def loadLDAKWeights(location) :
    fileName = location
    weights = list()
    
    counter = 0
    with open(fileName, "r") as id:
        for i in id:
            if counter > 0 : # weight file has header
                itmp = i.rstrip().split()
                weights.append( np.float64(itmp[1]) )
                
            counter = counter +1
            
    return ( weights )

def loadKinshipLocations(location) :
    fileName = location
    kinshipLocations = list()

    with open(fileName, "r") as id:
        for i in id:
            itmp = i.rstrip().split()
            kinshipLocations.append(itmp[0] )
 
            
    return ( kinshipLocations )


# loads the file that stores how many regions we have, and then loads each file in turn and adds their rsids into
def loadLDAKRegions(location) :
    fileName = location + "region_number"
    
    numRegions = -1
    with open(fileName, "r") as NFile:
        #itmp = NFile.rstrip().split()
        numRegions = np.int(NFile.readline())

    numRegions = numRegions +1 #  as we have the background region, which is 'region0'
    
    allRegions = list() # a list of lists
    for j in range(numRegions) :
        fileName = location + "region" + str(j)
     
        nextRegion = list()
        allRegions.append(nextRegion)
        with open(fileName, "r") as region:
            for i in region:
                nextRegion.append(i.rstrip().split()[0]) # each line in the file is the rsid

                
    return ( allRegions )


def loadLDAKRegionsDeltas(location) :# takes an reml file
    MAXDELTA = np.exp(10)
    allDeltas = list()
    with open(location, "r") as remlResults:
        for i in remlResults:
            if i.find("Her_") > -1 : # 
                itmp = i.rstrip().split() 
                h2 = np.float64( itmp[1]) # the 2nd item is the h2
                if h2 == 0 : delta  = MAXDELTA
                else : delta =   (1-h2) /h2
                allDeltas.append(delta)
                # h2 = Va/Vphe  ~ = Va/1   = Va 
                # Vphe = Va + Ve -> Ve = Vphe - Va  = Ve = 1 -h2
                # delta = Ve/Va == (1-Va) / Va  == (1-h2) / h2
                
    return(allDeltas)


# lines that contain the h2 results all contain "Her_"
# Her_K1 0.000000 0.000000
# Her_R1 0.000135 0.000773




    
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