# mdprop
Molecular dynamics library incorporating flexible construction of explicit 
integrators and potentials.

Explicit integrators are those that update the state of a dynamical system in a
stepwise manner. For example, velocity Verlet updates the velocity for half of a
timestep, then position for a timestep, then velocity again for half a timestep.
By decomposing an explicit integrator into a simple list of update functions and
corresponding coefficients we can build increasingly complex integrators with a
few lines of code.

This library is also similar to the atomic simulation environment (ASE) in that
potentials (and gradients/forces/Hessians) are constructed as stateful objects
that can be asked what the energy is given just coordinates. 
