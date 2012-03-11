# Carfac.py - Cochlear filter model based on Dick Lyons work.  This material taken from his Hearing book (to be published)
# Author: Al Strelzoff: strelz@mit.edu

# The point of this file is to extract some of the formulas from the book and practice with them so as to better understand the filters.
# The file has been written and tested for python 2.7

from numpy import cos, sin, pi, e, real,imag,arctan2


from pylab import figure, plot,loglog, title, axis, show

fs = 22050.0            # sampling rate


# given a frequency f, return the ERB
def ERB_Hz(f):
#    Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
    return 24.7 * (1.0 + 4.37 * f / 1000.0)


# ERB parameters
ERB_Q = 1000.0/(24.7*4.37)  # 9.2645
ERB_break_freq = 1000/4.37  # 228.833

ERB_per_step = 0.3333        

# set up channels

first_pole_theta = .78 * pi # We start at the top frequency.
pole_Hz = first_pole_theta * fs / (2.0*pi)  # frequency of top pole
min_pole_Hz = 40.0    # bottom frequency

# set up the pole frequencies according to the above parameters
pole_freqs = []     # empty list of pole frequencies to fill, zeroth will be the top
while pole_Hz >  min_pole_Hz:
    pole_Hz = pole_Hz - ERB_per_step * ERB_Hz(pole_Hz)
    pole_freqs.append(pole_Hz)
    
n_ch = len(pole_freqs)      # n_ch is the number of channels or frequency steps
print('num channels',n_ch)

# Now we have n_ch, the number of channels, so can make the array of filters by instantiating the filter class (see below)

# before we make the filters, let's plot the position of the frequencies and the values of ERB at each.

fscale = []
erbs = []

figure(1)
for i in range(n_ch):
    
    f = pole_freqs[i]        # the frequencies from the list
    ERB = ERB_Hz(f)       # the ERB value at each frequency
    fscale.append(f)
    erbs.append(ERB)
    
# plot a verticle hash at each frequency:
    u = []
    v = []
    for j in range(5):
        u.append(f)
        v.append(10.0 + float(j))

    plot(u,v)  
    
loglog(fscale,erbs)

title('ERB scale')



# This filter class includes some methods useful only in design.  They will not be used in run time implementation.
#  From figure 14.3 in Dick Lyon's book.
#  When translating to C++, this class will become a struct, and all the methods will be moved outside.


#########################################################The Carfac filter class#################################################################################

# fixed parameters
min_zeta = 0.12

class carfac():
    
    
    # instantiate the class (in C++, the constructor)
    def __init__(self,f):
        
        self.frequency = f
       
        theta = 2.0 * pi * f/fs
        r =  1.0 - sin(theta) * min_zeta
        a = r * cos(theta)
        c = r * sin(theta)
        h = sin(theta)
        g = (1.0 - 2.0 * a + r ** 2)/(1.0 - 2.0 * a + h * c + r ** 2)
      
       
        
        self.gh = g*h       # no need to repeat in real time
        
        # make all parameters properties of the class 
        self.a = a
        self.c = c
        self.r = r
        self.theta = theta
        self.h = h
        self.g = g
        
        
        # the two storage elements.  Referring to diagram 14.3 on p.263, z2 is the upper storage register, z1, the lower
        self.z1 = 0.0    
        self.z2 = 0.0    
      
        
        # frequency response of this filter
        self.H = []
        
       
        
        # the total  frequency magnitude of this filter including all the filters in front of this one
        self.HT = []       # this list will be filled by multiplying all the H's ahead of it together with its own (H)
        
        
    
    # execute one clock tick.  Take in one input and output one result. Execution semantics taken from fig. 14.3
    #  This execution model is not tested in this file.  Here for reference.  See the file Exec.py for testing this execution model.  This is the main run time method.
    def input(self,X):

        # recover the class definitions of these variables.  These statements below take up zero time at execution since they are just compiler declarations.
        # computation below is organized as some loads, followed by 3 2x2 multiply accumulates
        #  Note: this function is not exercised in this file and is here only for reference
        
        a = self.a
        c = self.c
        g = self.g
        gh = self.gh
        z1 = self.z1                    # z1 is the lower storage in fig. 14.3
        z2 = self.z2
        
        # calculate what the next value of z1 will be, but don't overwrite current value yet.   
        next_z1 = (a * z1) + (c * z2)   # Note:  it is a 2 element multiply accumulate. compute first so as not to have to do twice.
        # the output Y
        Y = g * X + gh * next_z1       # Note: organized as a 2 element multiply accumulate.
        
        #stores
        self.z2 = X + (a * z2) - (c * z1)        #Note: this is a 2 element multiply accumulate
        self.z1 = next_z1
        
        return Y                        # The output
    
    # complex frequency response of this filter at frequency w.  That is, what it contributes to the cascade
    # this method is used for test only.  It finds the frequency magnitude.  Not included in run time filter class.
    def Hw(self,w):

        a = self.a
        c = self.c
        g = self.g
        h = self.h
        r = self.r
        z = e ** (complex(0,w))         # w is in radians so this is z = exp(jw)
        return g * (1.0 + (h*c*z)/(z**2 - 2.0*a*z + r**2 ))      
        
    
    
######################################################End of Carfac filter class########################################################################
    
# instantiate the filters

# n_ch is the number of filters as determined above

Filters = []    # the list of all filters, the zeroth is the top frequency
for i in range(n_ch):
    f = pole_freqs[i]
    filter = carfac(f)  # note: get the correct parameters for r and h from Dick's matlab script.  Load them here from a table.
    Filters.append(filter)
    


# sweep parameters
steps = 1000

figure(2)   
title('CarFac individual filter frequency response')
# note:  the  scales are linear, not logrithmic as in the book


for i in range(n_ch):
    filter = Filters[i]
    # plotting arrays
    u = []
    v = []
    # calculate the frequency magnitude by stepping the frequency in radians
    for j in range(steps):
        
        w = pi * float(j)/steps     
        u.append(w)
        mag = filter.Hw(w)      # freq mag at freq w
        filter.H.append(mag)        # save for later use
        filter.HT.append(mag)       # will be total response of cascade to this point after we do the multiplication in a step below
        v.append(abs(mag))           # y plotting axis
        
      
    plot(u,v)


# calculate the phase response of the same group of  filters
figure(3)
title('Carfac individual filter Phase lag')


for i in range(n_ch):
    filter = Filters[i]
    
    u = []
    v = []
    for j in range(steps):
        x = pi * float(j)/steps
        
        u.append(x)
       
        mag = filter.H[j]
        phase = arctan2(-imag(mag),-real(mag))  - pi    # this formula used to avoid wrap around 
       
        v.append(phase)           # y plotting axis 
     
    plot(u,v)
    axis([0.0,pi,-3.0,0.05])
    
    
# calulate and plot cascaded frequency response    

figure(4)   
title('CarFac cascaded filter frequency response')


for i in range(n_ch-1):
   
    filter = Filters[i]
    next = Filters[i+1]
    
    
    u = []
    v = []
    for j in range(steps):
        w = pi * float(j)/steps    
        u.append(w)
        mag = filter.HT[j] * next.HT[j]
        next.HT[j] = mag
        v.append(abs(mag))

      
    plot(u,v)
   
  

# calculate and plot the phase responses of the cascaded filters

figure(5)
title('Carfac cascaded filter Phase lag')


for i in range(n_ch):

    filter = Filters[i]
    

    u = []
    v = []          # store list of phases
    w = []          # second copy of phases needed for phase unwrapping
    for j in range(steps):
        x = pi * float(j)/steps
        
        u.append(x)
        mag = filter.HT[j]
        a = imag(mag)
        b = real(mag)
        phase = arctan2(a,b) 
        
        v.append(phase)  
        w.append(phase)
        
    # unwrap the phase


    for i in range(1,len(v)):
        diff = v[i]-v[i-1]
        if diff > pi:
            for j in range(i,len(w)):
                w[j] -= 2.0 * pi
        elif diff < -pi:
            for j in range(i,len(w)):
                w[j] += 2.0 * pi
                
        else: continue
   
     
    plot(u,w)
    axis([0.0,pi,-25.0,0.05])
show()
    
