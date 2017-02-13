import numpy as np
import matplotlib.pyplot as plt

def GetEmptyFig():
  fig=plt.figure(frameon=False)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  return fig

#I = np.resize(np.arange(9),(100,100))
#fig = GetEmptyFig()
#plt.imshow(I,
#    interpolation="nearest", cmap=plt.get_cmap("Set1"))
#plt.savefig("../shaders/labels/rnd.png",figure=fig)
#plt.show()

I = np.random.rand(101,101)
for i in range(1,101):
  for j in range(1,101):
    while np.abs(I[i,j] - I[i-1,j]) < 0.1 or np.abs(I[i,j] - I[i,j-1]) < 0.1:
      I[i,j] = np.random.rand(1)
I = I[1::,1::]
fig = GetEmptyFig()
plt.imshow(I,
    interpolation="nearest", cmap=plt.get_cmap("jet"))
plt.savefig("../shaders/labels/rnd.png",figure=fig)
plt.show()
