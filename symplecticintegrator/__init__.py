from .integrators import *


_integrators_classes = {
    'symplectic': SymplecticIntegrator,
    'rk4': RungeKutta4Integrator
}


def integrator(hamiltonian, q_symbols, p_symbols, parameter_subs=None, itype='symplectic', order=1):
    """
    Creates an Integrator instance.

    Parameters
    ----------
    hamiltonian : Sympy Expr
        Hamiltonian symbolic expression.
    q_symbols : list of Sympy Symbols
        Position variables in the Hamiltonian.
    p_symbols : list of Sympy Symbols
        Momentum variables in the Hamiltonian.
    parameter_subs : dict, optional
        Substitutions of the free variables in the Hamiltonian. The keys are Sympy Symbols and the values are numbers.
    itype : {'symplectic', 'rk4'}, optional
        Type of integrator. One of

        ``symplectic``
            Symplectric integrator.

        ``rk4``
            Fourth-order Runge-Kutta method.

    order : {1, 2, 3, 4}, optional
        The order of the symplectic integrator (in case of ``itype='symplectic'``). One of

        ``1``
            First-order (Symplectic Euler method).

        ``2``
            Second-order (Verlet Integration).

        ``3``
            Third-order (Ruth).

        ``4``
            Fourth-order (Forest and Ruth).

    Returns
    -------
    Integrator instance.
    """
    cls = _integrators_classes[itype]
    if cls == SymplecticIntegrator:
        return cls(hamiltonian, q_symbols, p_symbols, parameter_subs=parameter_subs, order=order)
    return cls(hamiltonian, q_symbols, p_symbols, parameter_subs=parameter_subs)
