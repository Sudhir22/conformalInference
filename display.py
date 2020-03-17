import matplotlib.pyplot as plt
class Display:
    def __init__(self,x,y,label):
        self.x = x
        self.y = y
        self.label = label

    def timelines(self,y_index,xstart,xstop,color='b'):
        """Plot timelines at y from xstart to xstop with given color."""
        plt.vlines(y_index, xstart, xstop, color, lw=4)
        plt.hlines(xstart, y_index+0.03, y_index-0.03, color, lw=2)
        plt.hlines(xstop, y_index+0.03, y_index-0.03, color, lw=2)

    def plot_graph(self):
        for i in range(0,len(self.x)):
            self.timelines(self.y[i],self.x[i][0],self.x[i][1])
        ax = plt.gca()
        plt.xlabel("Test points")
        plt.ylabel(self.label)
        plt.savefig('Graphs/{}.png'.format(self.label))
