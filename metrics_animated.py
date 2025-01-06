import matplotlib.pyplot as plt
import matplotlib.animation as animation
    
class GifCreator:
    def __init__(self,title=None) -> None:
        self.title = title
        self.X=[]
        self.Y=[]

    def add_data(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        
    def get_color(self,i):
        color = [0.,0.,0.]
        color_idx = int(i/(len(self.X)/3))
        color[color_idx] = max((i-color_idx*(len(self.X)//3))/(len(self.X)//3), 0.5)
        return (color[0], color[2], color[1])
    
    def step(self, i):
        if i>=len(self.X):
            return self.line
        self.line, = self.ax.plot(self.X[i], self.Y[i], animated=True)
        
        if len(self.X)>10:
            self.line.set_color(self.get_color(i))
        else: # cannot get 3 colors (division by 0 risk)
            self.line.set_color((0, i/len(self.X), 0))
        # label every 10 epochs
        if i%10==0:
            self.line.set_label(f'epoch {i+1}')
        plt.legend()
        return self.line,
    
    def init(self):
        pass
    def create_gif(self, path_to_save):
        plt.clf()
        self.fig, self.ax = plt.subplots()
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        if self.title:
            plt.title(self.title)
        # self.line, = self.ax.plot(self.X[0], self.Y[0], animated=True)
        animated_fig = animation.FuncAnimation(self.fig, self.step,  frames=len(self.X)+100, init_func=self.init, interval=50, repeat=False)
        animated_fig.save(path_to_save)
if __name__=='__main__':
    import random as rd
    creator = GifCreator(title='test')
    
    for i in range(100):
        x = [j for j in range(10)]
        a  = rd.random()
        y = [i/100 + a  for j in range(10)]
        creator.add_data(x,y)
    creator.create_gif('test.gif')
    creator.create_gif('test2.gif')