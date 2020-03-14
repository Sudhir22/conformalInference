from scipy.stats import wilcoxon
import csv

class WilcoxonTest:

    def __init__(self,x):
        self.x = x
        self.result = dict()

    def test(self):
        for key,value in self.x.items():
            u,p = wilcoxon(self.x[key],alternative='greater')
            self.result[key] = p


    def sort(self,filename):
        self.result = {k:v for k,v in sorted(self.result.items(), key=lambda item : item[1])}
        '''with open(filename,'w') as f:
            w = csv.writer(f)
            w.writerows(self.result.items())
            '''
