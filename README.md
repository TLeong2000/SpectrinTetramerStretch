# SpectrinTetramerStretch
 Measure the force exerted on the endpoints of a spectrin tetramer when kept a constant distance apart at thermal equilibrium

 Let * refer to the Alpha/Beta gene that encodes a spectrin. 
 
 Proto*.txt refers to the an initial prototype structure for the spectrin filament. The endpoint particles have the same relative position, while the internal particles have randomized positions. These prototype structures then undergo an energy minimization, with the endpoint particles kept stationary. After re-centering the center-of-geometry, the coordinates of the minimized structure is then saved as the production model to be used for NVT stretching, as production*.txt

 *Coeffs.data is a LAMMPS data file that contains the per-type masses and bond/angle coeffs for *.
