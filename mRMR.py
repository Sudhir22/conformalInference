import pymrmr


class MRMR:
    def __init__(self,data,k):
        self.data = data
        self.k = k
        self.method = 'MID'

    '''
    MrMR algoithm to find the top 'k' features

    '''

    def findBestFeatures(self):
        features = pymrmr.mRMR(self.data,self.method,self.k)
        return features


    '''
    Function to iterate over a dictionary

    '''
    def iterateDict(self,dictionary):
        for key,value in dictionary.items():
            print(key)
            print(value,"\n")
