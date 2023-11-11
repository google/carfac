# CARFAC v2 Release Notes

The update we refer to as CARFAC v2 (or as lyon2023, as opposed to lyon2011, in the Auditory Model Toolbox adaptation) includes the following changes in the Matlab version (and corresponding changes in the new Numpy and Jax versions, but not yet in the C++):

1. High-pass filter at BM output.
> We moved the 20 Hz highpass filter from IHC to CAR, to roughly model helicotrema reflection that shorts out DC BM response.  There is still a fair amount of low-frequency quadratic distortion in the BM output, but not DC.  The IHC output (the NAP) is not affected by this change, but the BM output is: recomputing the book's Figure 17.7 shows that the stripe at 0 frequency disappears.

2. Bug fix: stop parameter changes in open-loop mode.
> Previously, if the mode was changed to open-loop while the model was ramping the AGC feedback-controlled filter parameters, the ramping that was intended to interpolate to the new value would continue ramping, extrapolating beyond the intended value.  With this bug fix, the ramping is stopped (when calling CARFAC_Run_Segment) in open-loop mode.

3. Linear CAR option.
> Optionally make the OHC nonlinear function just a 1, to make the cascade of asymmetric resonators (CAR) be a linear filterbank, non-distorting if open loop, e.g. for characterizing transfer functions in tests.  If the CAR is not run open loop, there will still be parameter adaptation, yielding compression and some odd-order distortion.

4. IHC model change to support receptor potential output.
> We removed the previous (not used by default) two-capacitor inner hair cell (IHC) model, and replaced it by a new one that allows interpreting one of the capacitor voltages as a receptor potential.  This new two-cap version is the default; it results in rather minor differences in the IHC output (including a somewhat reduced peak-to-average ratio in the NAP output), and via AGC feedback also very tiny differences in the BM output.

5. Interface for selecting IHC model type.
> The function CARFAC_Design now takes an optional last arg, keyword 'one_cap' or 'just_hwr' to change from the default 'two_cap' inner hair cell model.  This makes it easier to get back to the old one-cap IHC model if desired, and easier to make tests that compare them.

6. Delay per stage for better parallelization.
> We optionally include an extra sample delay per stage, to make the CAR step parallelize better, avoiding the "ripple" iteration.  Off by default, but a useful option for speeding up execution on parallel architectures.

7. Simplification of stage-gain computation.
> The stage-gain update required evaluation of a ratio of polynomials; it has been replaced by a quadratic polynomial approximation, somewhat more accurate than the one suggested in the book Figure 16.3.

8. AGC spatial smoothing simplification.
> We simplified the AGC spatial smoothing to just the 1-iteration 3-tap spatial FIR filter structure, which was the default and is the one illustrated in the book Figure 19.6.  If more spatial smoothing is requested than this filter can handle, the AGC decimation factor will be reduced instead of changing the filter architecture.  This simplifies ports to FPGA, Jax, etc.

9. Outer hair cell health.
> Provision for modeling sensorineural hearing loss via the OHC_health parameter vector (a health value between 0 and 1 per channel, multiplying the relative undamping, defaulting to 1).

10. Extensive tests.
> We have added test routines to the open-source code.  We have used these to help us keep the Matlab, Numpy, and Jax versions synchronized, and to facilitate updating the C++ version as well, if someone wants to work on that.

