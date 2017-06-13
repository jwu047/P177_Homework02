
# coding: utf-8

# # Homework 2: Integration at Work
# Physics 177, Spring 2017  (Prof. Tanedo)  
# **Due**: Tuesday, April 25
# 
# Your Name  
# 
# <span style="color:red">**Revised** (April 23): Fixed: problem numbering. *I've been informed that some students feel that this homework is too much work. I am reducing the homework from the original, please see the notes in red below.*</span>  
# 
# <span style="color:blue">
# The revisions are as follows:  
# **Problem 1.** You may write `integrate_HW` using the trapezoidal rule instead of Simpson's rule.  
# **Problem 2.** Only parts A and B are required.  
# **Problem 3.** Now completely optional.  
# **Problem 4.** Unchanged.
# </span>
# 
# <span style="color:blue">
# **Important**: For any of the optional problems that you choose to skip, you **must** explicitly explain why you were unable to complete the problem. For example, "I tried to code up Simpson's rule but I could not figure out how to have the loop treat odd and even steps differently." If you just write, "*It was too hard,*" then you will lose points (and probably self-respect).
# </span>
# 
# 

# ## 1. Simpson's Rule, Again
# 
# <span style="color:red">**Revised** (April 23): you may chose to do this using the trapezoidal rule. If you do, please explain why you were unable to code Simpson's rule.</span>
# 
# Write another integrator function using **Simpson's rule**. You may adapt your code from Homework 1. The integrator should take the following arguments:
# * A function, $f$
# * Two real numbers giving the limits of integration, $a$ and $b$
# * The number of steps $N$ to sample
# 
# This differs from Homework 1, where you specified the $\Delta x$ rather than $N$. There are different ways of implementing Simpson's rule. Here's one example from (5.9) of Newman's *Computational Physics*:
# 
# $\displaystyle I(a,b) = \frac{1}{3}\Delta x\left[ f(a) + f(b) + 4 \sum_{k \text{ odd}}^{N-1} f(a+k\Delta x) + 2 \sum_{k \text{ even}}^{N-2} f(a+k\Delta x) \right]$ ,
# 
# where the step size is $\Delta x = (b-a)/N$.
# 
# You can fill in the following function stub:

# In[12]:

def integrate_HW2(f,a,b,N):
    """
    Integrate f from a to b using Simpson's rule with N steps
    """
    
    # Your code goes here!
    h = (b-a)/N
    # h is delta x, the step size
    k = 0.0
    x = a + h 
    
    #For odds
    for i in range(1, int((N/2)-1)):
        x += 2*h
        k += 4*f(x)
        
    # For evens:
    for i in range(1, int((N/2))):
        x += 2*h
        k += 2*f(x)
                   
    return (h/3)*(f(a)+f(b)+k) # Simposon's Rule


# Now test this on a nice function:

# In[13]:

def test_function(x):
    return 4.0*x**3

integrate_HW2(test_function,0,1,100)


# ## 2. Diffraction Limit of a Telescope
# [Adapted from Exercise 5.4 from *Computational Physics* by Newman]
# 
# ### Discussion
# Our ability to resolve detail in astronomical observations is limited by the diffaction of light in our telescopes.  Light from stars can be treated effectively as coming from a point source at infinity. When such light, with wavelength $\lambda$, passes through the circular aperture of a telescope (which we'll assume to have unit radius) and is focused by the telescope in the focal plane, it produces not a single dot, but a circular diffraction patter consisting of a central spot surrounded by a series of concentric rings. The intensity of the light in this diffraction pattern is given by
# 
# $\displaystyle I(r) = \left(\frac{J_1(kr)}{kr}\right)^2 \ ,$
# 
# where $r$ is the distance in the focal plane from the center of the diffraction pattern, $k = 2\pi/\lambda$ and $J_1(x)$ is a Bessel function. The Bessel functions $J_m(x)$ are given by
# 
# $\displaystyle J_m(x) = \frac{1}{\pi}\int_0^\pi \cos(m\theta - x \sin \theta) \, d\theta\ ,$ 
# 
# where $m$ is a nonnegative integer and $x\geq 0$. 

# ### A. Coding the Bessel function

# Write a Python function that calculates the value of $J_m(x)$ using Simpson's rule with $N=1000$ points. 

# In[9]:

from math import pi as pi
from math import sin as sin
from math import cos as cos
"""
mth Bessel function of the first kind evaluated at real number x,  
calculated using an integral representation with Simpson's rule and 1000 steps
"""
# Your code goes here!
    
def f(m,x,theta):
    return cos(m*theta - x*sin(theta))

#Simpson's rule with N = 1000 points
N = 1000

def J(m,x):
    
    #limits
    a = 0.0
    b = pi
    
    #step size
    h = (b - a) / N
    
    # Use the integrate_HW2 function from Problem 1
    
    #f(a)+f(b)
    k = f(m, x, a) + f(m, x, b)
    
    for i in range(1, N):
        theta = a + i*h
        #For odds
        for i in range(1, int((N/2)-1)):
            x += 2*h
            k += 4*f(m, x, theta)
        
        #For evens
        for i in range(1, int((N/2))):
            x += 2*h
            k += 2*f(m, x, theta)
        # (h/3) to get the integral, (1/pi) from Bessel function, k for the total
        return (1/pi)*(h/3)*k



# ### B. Plotting these Bessel functions
# Plot the Bessel functions $J_0$, $J_1$, and $J_2$ from $x=0$ to $x=20$. 
# 
# Make sure that you have installed `matplotlib`. You can do this using `pip` at the command line: (`pip` vs `pip3` depends on your installation)  
# `pip3 install matplotlib`
# 
# Note: if the plot doesn't show up in this notebook, you may need to run  
# `%matplotlib inline`
# in the notebook
# 
# Follow the template at the bottom of https://github.com/Physics177-2017/Lec05-numpyintro/blob/master/Numpy%20Introduction.ipynb

# In[31]:

import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline ## uncomment if necessary


myWeirdSine = np.vectorize(J)

x = np.linspace(0,20,100)
x = np.linspace(0,20,1000)

plt.plot(x, myWeirdSine(0,x), label='$m=0$')
plt.plot(x, myWeirdSine(1,x), label='$m=1$')
plt.plot(x, myWeirdSine(2,x), label='$m=2$')
plt.title("Bessel function")
plt.xlabel('x')
plt.ylabel('J')
plt.legend()
plt.show()



# In[59]:

# Create a Numpy array of sampling points
x = np.linspace(0, 20, 100)

# Create a "vectorized" version of J(m,x) that 
# accepts Numpy arrays as arguments
Jnp = np.vectorize(J)

plt.plot(x, Jnp(0, x), label='$J_0(x)$')
         # FILL THIS IN

# FILL IN THE REST
# Make sure to include plot labels

plt.title("Bessel Functions")
plt.xlabel('x')
#unable to add plt.ylabel
# 'str' object is not callable
plt.ylabel('J_0(x)')
plt.legend()
plt.show()


# ### C. Comparing to Sines
# 
# <span style="color:red">**Revised** (April 23): this problem is now optional. If you choose to skip it, explain why you were unable to code these functions and plot them in comparison to the integral expressions.</span>
# 
# Over the same range as before, plot $J_0(x)$, $J_1(x)$ (as you did above) and the following functions:
# 
# $\displaystyle f_0(x) = \sqrt{\frac{2}{\pi x}} \cos\left(\pi - \frac{\pi}{4}\right)$  
# $\displaystyle f_1(x) = \sqrt{\frac{2}{\pi x}} \cos\left(\pi - \frac{3\pi}{4}\right)$
# 
# *Remark*: the more general formula is:  
# $\displaystyle f_m(x) = \sqrt{\frac{2}{\pi x}} \cos\left(\pi - \frac{(2m+1)\pi}{4}\right)$,  
# this is supposed to be a good approximation of $J_m(x)$ for large $x$.

# In[65]:

# No quick start here. 
# Write your own functions f0(x) and f1(x),
# then follow the Problem 2B template to plot f0, f1, J0, and J1

import math

def F(m,x):
    return math.sqrt(2/(pi*x))*cos(pi-((2*m+1)*pi/4)*(x))

x = np.linspace(0, 20, 100)

# Create a "vectorized" version of J(m,x) that 
# accepts Numpy arrays as arguments
Jnp = np.vectorize(J)
Fnp = np.vectorize(F)

plt.plot(x, Jnp(0, x), label='$J_0(x)$')
plt.plot(x, Jnp(1, x), label='$J_1(x)$')
plt.plot(x, Fnp(0, x), label='$F_0(x)$')
plt.plot(x, Fnp(1, x), label='$F_1(x)$')

plt.title("Bessel Functions")
plt.xlabel('x')
plt.ylabel('J_0(x)')
plt.legend()
plt.show()


# ### D. Plotting Diffraction Patterns
# 
# <span style="color:red">**Revised** (April 23): this problem is now optional. If you choose to skip it, explain why you were unable to plot the diffraction pattern following the template provided.</span>
# 
# Use the equation for $I(r)$ above to plot the diffraction pattern seen, for example, by a telescope when viewing a point source at the origin. Your plot should look like an **Airy disk**:
# 
# ![title](https://upload.wikimedia.org/wikipedia/commons/1/14/Airy-pattern.svg)
# 
# Image (and discussion) from:
# https://en.wikipedia.org/wiki/Airy_disk
# 
# For our purposes, use $\lambda = 500~\text{nm}$. Your plot should cover values of $r$ from zero up to $1~\mu\text{m}$. Be careful with units! Numerical functions don't know how to handle units, so you should make sure you're consistent everywhere. I suggest using the Numpy functions **meshgrid** and **pcolormesh**, as demonstrated below.
# 

# In[20]:

side = np.linspace(-10,10,50)
X,Y = np.meshgrid(side,side)
Z = np.sin(X**2 + Y**2)
plt.pcolormesh(X,Y,Z)
plt.title("Silly Plot")
plt.show()


# Try to plot the Airy disk using a variation of the above code, where `Z` is given by $I(r)$, which you have to code in appropriately. 

# In[77]:

# Your code here!
side = np.linspace(0,1,50)
X,k = np.meshgrid(side,side)
#unsure of how to plug in kr for x, "J() missing 1 required positional argument: 'x' "
x = (500*10**-9)*(1*10**-6)
Z = (J(x)/ x)**2
plt.pcolormesh(X,Y,Z)
plt.title("Silly Plot")
plt.show()


# You should find something that looks like a Gaussian, but without the diffraction patterns being immediately obvious. This is because the intensity near the center is so much brighter than the intensity of the Airy disks, so the disks are hard to see using the default plot parameters.  
# 
# The `vmax` option tells you what the maximum contour value should be; anything above this value will be mapped to the same "large value" color.

# In[79]:

plt.pcolormesh(X,Y,Z, vmax=0.1)
plt.title("Silly Plot")
plt.show()

# 'Z' is not defined, cannot change vmax to see results


# Play with `vmax` to try to resolve the Airy disks. I got mine to work for `vmax=0.01`.
# 
# 
# ### X. Extra credit
# 
# Refer to the `matplotlib` documentation (or any other resource) to play with the plot options. Change the aspect ratio so that the x- and y- spacings are equal... in other words: make circles look like circles, not ovals. Also play with the color schemes. Can you make a plot that looks like something an astronomer would see (white for areas where there are photons, black for areas where there are no photons). Can you put a plot legend that quantifies the intensity of the plot?

# ## 3. Gaussian Quadrature
# 
# <span style="color:red">**Revised** (April 23): this entire problem is now optional. You still have to explain why you were unable to do it if you choose to skip it.</span>
# 
# Download the implementation of **Gaussian Quadrature** from Mark Newman's *Computational Physics* website, or use the copy of `gaussxw.py` included in this GitHub repo. 
# 
# A sample program is provided below. Note how the package is called using  
# `from gaussxw import gaussxw`  
# ... this assumes that `gaussxw.py` is in the same directory as this notebook.
# 
# Here's how the package works:

# In[81]:

from gaussxw import gaussxw

def f(x):
    return x**4 - 2*x + 1

N=3    # number of steps
a=0.0  # lower limit of integration
b=2.0  # upper limit of integration

x,w = gaussxw(N)
# gaussxw(N) returns the sample points x and weights w
# for the reference integral (-1,1)

# Now define x' and w' to be the sample points and weights
# for the interval (a,b). We derived this transformation
# in lecture 5.
xp = 0.5*(b-a)*x + 0.5*(b+a)
wp = 0.5*(b-a)*w

# Now estimate the integral of f by summing
# the sample points and the weights
total = 0.0
for k in range(N):
    total += wp[k]*f(xp[k])
    
print(total)


# To better understand this, actually print the inputs and outputs

# In[38]:

print(x)
print(w)


# ### A. Plotting the weights
# 
# <span style="color:red">**Revised** (April 23): if you choose to skip this problem, please explain why it was too difficult.</span>
# 
# Create a histogram (or scatter plot) of the sample points and weights that come from the Gaussian quadrature procedure as encoded in `gaussxw`. You should have a plot whose y-axis is the weight, and whose x-axis is the sample position. 
# 
# Note: you will have to convert the Python arrays that `gaussxw` outputs into Numpy arrays. You can do this with `np.array(x)` where `x` is a Python array.
# 
# Make separate plots for $N = 5, 110, 100$.

# In[85]:

from gaussxw import gaussxw
from matplotlib import pyplot as plt
import numpy as np

gaussian_numbers = gaussxw(5)
gaussian_numbers = gaussxw(110)
gaussian_numbers = gaussxw(100)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
#ylabel is not showing again... " 'str' object is not callable"
#There is one giant orange line in the center, I am unsure of why this is.
plt.ylabel("Frequency")

fig = plt.gcf





# ### B. Problems 1A and 1B using Gaussian Quadrature
# 
# <span style="color:red">**Revised** (April 23): if you choose to skip this problem, please explain why it was too difficult.</span>
# 
# Redo the steps in Problem 1A and Problem 1B using **Gaussian quadrature** rather than Simpson's rule. Use `gaussxw` as shown above. 
# 
# In order to do this, you'll need a way a way to get $J_m(x)$ samples. I suggest using the built in "Bessel function of the first kind" function in `scipy`.
# 
# You may need to install `scipy` using the following command at the terminal  
# `pip3 install scipy`

# In[55]:

from scipy.special import jv as jv  # jv is the J_m from above

print(jv(1,1)) ## compare this with J(1,1) from Problem 1


# In[56]:

# Your code here!
# Code the Bessel functions as the integral representation shown above
# ... except now use jv(m,x) to sample the Bessel functions
# ... and focus on the range x in (0,20)
# ... and use Gaussian quadrature to do the integration

# Then plot your Bessel functions, confirm that they match what you got in Problem 1


# ### X Extra Credit: small sampling
# 
# Your Bessel function plots in Problem 1 and 2 should look the same. See what happens if we become "cheap" with our number of samplings. Plot the Bessel functions using Simpsons rule and using Gaussiand quadrature for low values of $N$. 

# ## 4. Simpson's Rule Error from HW1
# 
# If you did things correctly in HW1, you will have noticed that your errors using Simpsons rule for a function of the form $f(x) = ax^2$ were strange: they were small and didn't scale as you'd expect. 
# 
# ### A. Explain why
# 
# In your own words, explain why this happened. What would you expect from a function of the form $f(x) = ax^3$?
# 
# **Hint**: in class we argued that the approximation error using the trapezoidal rule was  
# $\displaystyle \epsilon = \frac{1}{12} \delta x^2 \left[ f'(a) - f'(b) \right]$.  
# 
# 
# ### X. Extra credit
# 
# Following Chapter 5.2 of Newman's *Computational Physics*, derive the factor of $1/12$ in the above expression for $\epsilon$. (This derivation is also the lecture 5 notes.) Derive the analogous approximation error for Simpson's rule. It has the form:
# 
# $\displaystyle \epsilon = \frac{1}{?} \Delta x^4 \left[ \cdots ? \cdots \right]$ .

# *Your answer here*

# ## X Extra Credit: animations
# 
# Refer to: https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# (Or any other matplotlib animation tutorial)
# 
# Make an animated phase-shifted $J_0(x)$, analogous to the phase-shifted sine wave animation at the beginning of the above tutorial.

# In[1]:

# Your work here

