from .integrators import *


_integrators_classes = {
    'symplectic': SymplecticIntegrator,
    'rk4': RungeKutta4Integrator
}


def integrator(hamiltonian, q_symbols, p_symbols, parameter_subs=None, itype='symplectic', order=1):
    cls = _integrators_classes[itype]
    if cls == SymplecticIntegrator:
        return cls(hamiltonian, q_symbols, p_symbols, parameter_subs=parameter_subs, order=order)
    return cls(hamiltonian, q_symbols, p_symbols, parameter_subs=parameter_subs)
