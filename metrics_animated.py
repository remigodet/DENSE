import matplotlib.pyplot as plt
import matplotlib.animation as animation
    
class GifCreator:
    def __init__(self,title=None) -> None:
        self.fig, self.ax = plt.subplots()
        self.title = title
        self.X=[]
        self.Y=[]

    def add_data(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        
    def step(self, i):
        self.line, = self.ax.plot(self.X[i], self.Y[i], animated=True)
        self.line.set_color((0, i/len(self.X), 0))
        if i%10==0:
            self.line.set_label(f'epoch {i+1}')
        plt.legend()
        return self.line,
    
    def create_gif(self, path_to_save):
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        if self.title:
            plt.title(self.title)
        self.line, = self.ax.plot(self.X[0], self.Y[0], animated=True)
        animated_fig = animation.FuncAnimation(self.fig, self.step,  frames=10, interval=200, repeat_delay=10,)
        animated_fig.save(path_to_save)
    
if __name__=='__main__':
    
    creator = GifCreator(title='test')
    
    for i in range(10):
        x = [j for j in range(10)]
        y = [i/10 for j in range(10)]
        
        creator.add_data(x,y)
    creator.create_gif('test.gif')