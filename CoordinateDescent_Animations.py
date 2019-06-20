import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import os

x = np.array([1,3,6])
y = np.array([6,10,16])

# The plot: LHS is the data, RHS will be the cost function.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
ax[0].scatter(x, y, marker='x', s=40, color='k')

def cost_func(theta0, theta1):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    ind = np.random.choice([0,1,2])
    x1, y1 = x[ind:ind+1], y[ind:ind+1]
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((y1-hypothesis(x1, theta0, theta1))*2, axis=2)*1.0/2

def hypothesis(x, theta0, theta1):
    """Our "hypothesis function", a straight line."""
    return theta0 + theta1*x

# First construct a grid of (theta0, theta1) parameter pairs and their
# corresponding cost function values.
theta0_grid = np.linspace(-1,4,101)
theta1_grid = np.linspace(-5,5,101)
J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
                   theta1_grid[:,np.newaxis,np.newaxis])

# A labeled contour plot for the RHS cost function
X, Y = np.meshgrid(theta0_grid, theta1_grid)
contours = ax[1].contour(X, Y, J_grid, 30)
ax[1].clabel(contours)
theta0_true=4
theta1_true=4
# The target parameter values indicated on the cost function contour plot
ax[1].scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])

# Take N steps with learning rate alpha down the steepest gradient,
# starting at (theta0, theta1) = (0, 0).
N = 50
alpha = 0.12
theta = [np.array((0,0))]
m=3
J = [cost_func(*theta[0])[0]]
for j in range(N-1):
    last_theta = theta[-1]
    this_theta = np.empty((2,))
    this_theta[0] = last_theta[0] - alpha / m * np.sum(
                                    (hypothesis(x, *last_theta) - y))
    this_theta[1] = last_theta[1] - alpha / m * np.sum(
                                    (hypothesis(x, *last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_func(*this_theta))


# Annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# Also plot the fit function on the LHS data plot in a matching colour.
colors = ['b', 'g', 'm', 'c', 'orange']
#ax[0].plot(x, hypothesis(x, *theta[0]), color=colors[0], lw=2,
          # label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[0]))

frames = []

imgflag = False
if not os.path.exists('3a_iter'):
    os.mkdir('3a_iter')
    
os.chdir('3a_iter')


if os.path.exists('iteration'+str(N-1)+'.png'):
    imgflag = True

for j in range(1,N):
    if not imgflag:
        ax[1].annotate('', xy=theta[j], xytext=theta[j-1],
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
        ax[0].clear()
        ax[0].scatter(x, y, marker='x', s=40, color='k')
        ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[2], lw=2,
               label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
        fig.savefig(r'iteration'+str(j)+'.png')
    img = plt.imread(r'iteration'+str(j)+'.png')
    frames.append(img)
            
        
        
        
#ax[1].scatter(*zip(*theta), c=colors, s=40, lw=0)

# Labels, titles and a legend.
ax[1].set_xlabel(r'$\theta_0$')
ax[1].set_ylabel(r'$\theta_1$')
ax[1].set_title('Cost function')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Data and fit')
axbox = ax[0].get_position()
# Position the legend by hand so that it doesn't cover up any of the lines.
ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
             fontsize='small')


##ani = animation.FuncAnimation(fig, run, [0]*N, blit=True, interval=10,
##    repeat=False)
fig2 = plt.figure()
ims = []
for i in range(len(frames)):
    image = frames[i]
    im = plt.imshow(image, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig2, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()


'''
References:
https://matplotlib.org/2.1.2/gallery/animation/dynamic_image2.html
'''
