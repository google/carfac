# Carfac.py - Cochlear filter model based on Dick Lyons work.  This material taken from his Hearing book (to be published)
# Author: Al Strelzoff

from numpy import cos, sin, tan, sinh, arctan, pi, e, real,imag,arccos,arcsin,arctan2,log10,log

from pylab import figure, clf, plot,loglog, xlabel, ylabel, xlim, ylim, title, grid, axes, axis, show

fs = 22050.0            # sampling rate
Nyq = fs/2.0            # nyquist frequency


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

figure(0)
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
        
      
        h = c
        
        g = 1.0/(1.0 + h * r * sin(theta) / (1.0 - 2.0 * r * cos(theta) + r ** 2))
        
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
        
        
        a = self.a
        c = self.c
        h = self.h
        g = self.g
        z1 = self.z1                    # z1 is the lower storage in fig. 14.3
        z2 = self.z2
        
        # calculate what the next value of z1 will be, but don't overwrite current value yet.   
        next_z1 = (a * z1) - (c * z2)   # Note: view this as next_z1 = a*z1 + (-c*z2) so that it is a 2 element multiply accumulate
        # the output Y
        Y = g * (X + h * next_z1)       # Note: reorganize this as Y = g*X + (g*h) * next_z1     g*h is a precomputed constant so then the form is a 2 element multiply accumulate.
        
        #stores
        z2 = (a * z2) + (c * z1)        #Note: this is a 2 element multiply accumulate
        z1 = next_z1
        
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
        return g * (1.0 + (h*c*z)/(z**2 - 2.0*a*z + r**2 ))     # from page ?? of Lyon's book.   
        
        
    # Note: to get the complex frequency response of this filter at frequency w, get Hw(w) and then compute arctan2(-imag(Hw(w))/-real(Hw(w)) + pi
      
    
    
    
    
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

sum = []    # array to hold the magnitude sum
for i in range(steps): sum.append( 0.0 )

figure(1)   
title('CarFac  frequency response')

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
        v.append(real(mag))           # y plotting axis
        sum[j]+= mag    
        
      
    plot(u,v)
    


figure(2)
title('Summed frequency magnitudes')
for i in range(steps): sum[i] = abs(sum[i])/n_ch
plot(u,sum)

# calculate the phase response of the same group of  filters
figure(3)
title('Filter Phase')


for i in range(n_ch):
    filter = Filters[i]
    
    u = []
    v = []
    for j in range(steps):
        x = float(j)/Nyq
        
        u.append(x)
       
        mag = filter.H[j]
        phase = arctan2(-imag(mag),-real(mag)) + pi     # this formula used to avoid wrap around
       
        v.append(phase)           # y plotting axis 
     
    plot(u,v)

    
    
# calulate and plot cascaded frequency response and summed magnitude    
sum = []    # array to hold the magnitude sum
for i in range(steps): sum.append( 0.0 )


figure(4)   
title('CarFac Cascaded  frequency response')


for i in range(n_ch-1):
   
    filter = Filters[i]
    next = Filters[i+1]
    
    
    u = []
    v = []
    for j in range(steps):
        u.append(float(j)/Nyq)
        mag = filter.HT[j] * next.HT[j]
        filter.HT[j] = mag
        v.append(real(mag))
        sum[j]+= mag    

      
    plot(u,v)
   
  

figure(5)
title('Summed cascaded frequency magnitudes')
for i in range(steps): sum[i] = abs(sum[i])/n_ch
plot(u,sum)

# calculate and plot the phase responses of the cascaded filters

figure(6)
title('Filter cascaded Phase')


for i in range(n_ch):
    filter = Filters[i]
    

    u = []
    v = []
    for j in range(steps):
        x = float(j)/Nyq
        
        u.append(x)
        mag = filter.HT[j]
        phase = arctan2(-imag(mag),-real(mag)) + pi
          
        v.append(phase)           # y plotting axis 
        
        

      
    plot(u,v)

show()
    
