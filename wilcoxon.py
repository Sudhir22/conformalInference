import scipy
import csv

class WilcoxonTest:

    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.result = dict()

    def test(self):
        for key,value in self.x.items():
            u,p = scipy.stats.wilcoxon(self.x[key],self.y)
            self.result[key] = p


    def sort(self,filename):
        self.result = {k:v for k,v in sorted(self.result.items(), key=lambda item : item[1])}
        with open(filename,'w') as f:
            w = csv.writer(f)
            w.writerows(self.result.items())
