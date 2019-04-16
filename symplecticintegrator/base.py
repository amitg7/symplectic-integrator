import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


class Integrator:
    """
    Base class of integrators.
    """
    def single_step_func(self, q_curr, p_curr, dt):
        """
        Evolve q and p vectors one time step.

        This function should be override in case of inheriting this (Integrator) class.

        Parameters
        ----------
        q_curr : array
            Current q vector.
        p_curr : array
            Current p vector.
        dt : float
            Duration of time step.

        Returns
        -------
        q_next : array
            The q vector after one time step.
        p_next : array
            The p vector after one time step.
        """
        q_next = q_curr
        p_next = p_curr
        return q_next, p_next

    @staticmethod
    def _assert_parameters(hamiltonian, q_symbols, p_symbols, parameter_subs):
        free_symbols = hamiltonian.free_symbols - set(q_symbols + p_symbols) - set(parameter_subs.keys())
        if len(free_symbols) > 0:
            raise ValueError('Not all the parameters have values. Please assign the value of {}'.format(str(list(free_symbols))))

    @staticmethod
    def _apply_func(func, q, p):
        y = np.concatenate([q, p], axis=0)
        y_splitted = np.split(y, 2, axis=0)
        return np.reshape(func(*y_splitted), (q.shape[0], -1))

    def _equations_of_motion(self):
        dq_dt_expr = sp.Matrix([sp.diff(self.hamiltonian, pi).simplify() for pi in self.p_symbols])
        dp_dt_expr = sp.Matrix([- sp.diff(self.hamiltonian, qi).simplify() for qi in self.q_symbols])
        dq_dt_func = sp.lambdify(self.y_symbols, dq_dt_expr.subs(self.parameter_subs))
        dp_dt_func = sp.lambdify(self.y_symbols, dp_dt_expr.subs(self.parameter_subs))
        def dq_dt(q, p): return self._apply_func(dq_dt_func, q, p)
        def dp_dt(q, p): return self._apply_func(dp_dt_func, q, p)
        return dq_dt, dp_dt

    def __init__(self, hamiltonian, q_symbols, p_symbols, parameter_subs=None):
        """
        Parameters
        ----------
        hamiltonian : Sympy Expr
            Hamiltonian symbolic expression
        q_symbols : list of Sympy Symbols.
            Position variables in the Hamiltonian.
        p_symbols : list of Sympy Symbols.
            Momentum variables in the Hamiltonian.
        parameter_subs : dict
            Substitutions of the free variables in the Hamiltonian. The keys are Sympy Symbols and the values are numbers.
        """
        super().__init__()
        self.parameter_subs = parameter_subs if parameter_subs is not None else {}
        self._assert_parameters(hamiltonian, q_symbols, p_symbols, self.parameter_subs)
        self.hamiltonian = hamiltonian
        self.q_symbols = q_symbols
        self.p_symbols = p_symbols
        self.y_symbols = [q_symbols, p_symbols]
        self.energy_func = sp.lambdify(self.y_symbols, hamiltonian.subs(self.parameter_subs))
        self.dq_dt, self.dp_dt = self._equations_of_motion()

    def evolve_iter(self, q_initial, p_initial, t_range):
        """
        Returns iterator of the evolution of q and p.

        Parameters
        ----------
        q_initial : array
            The q vector at time zero.
        p_initial : array
            The p vector at time zero.
        t_range : array_like
            Array of time points. Must start in zero and in ascending order.

        Yields
        ------
        generator
            The temporal q and p vectors.

        """
        step_dt = np.diff(t_range)
        q = np.copy(q_initial)
        p = np.copy(p_initial)
        yield q, p
        for step in np.arange(np.size(step_dt)):
            q, p = self.single_step_func(q, p, step_dt[step])
            yield q, p

    def evolve(self, q_initial, p_initial, t_range):
        """
        Evolve q and p in time.

        Parameters
        ----------
        q_initial : array
            The q vector at time zero.
        p_initial : array
            The p vector at time zero.
        t_range : array_like
            Array of time points. Must start in zero and in ascending order.

        Returns
        -------
        q_time : array
            Trajectory in time of q vector. The third axis is the time dimension.
        p_time : array
            Trajectory in time of p vector. The third axis is the time dimension.
        """
        q_time, p_time = np.stack(list(self.evolve_iter(q_initial, p_initial, t_range)), axis=2)
        self.t_range = t_range
        self.q_time = q_time
        self.p_time = p_time
        return q_time, p_time

    def energy_time(self):
        return self.energy_func(self.q_time, self.p_time)

    def plot(self, indices=None):
        if isinstance(indices, int):
            _indices = [indices]
        elif isinstance(indices, list):
            _indices = indices
        else:
            _indices = range(self.q_time.shape[0])
        for i in _indices:
            plt.plot(self.q_time[i, ...], self.p_time[i, ...], label=r'$({},{})$'.format(self.q_symbols[i], self.p_symbols[i]))
        plt.xlabel('position', fontsize=14)
        plt.ylabel('momentum', fontsize=14)
        plt.legend()

    def plot_energy(self):
        plt.plot(self.t_range, self.energy_time())
        plt.xlabel(r'$t$', fontsize=14)
        plt.ylabel(r'$E(t)$', fontsize=14)

    def plot_energy_error(self):
        initial_energy = self.energy_time()[0,...]
        plt.plot(self.t_range, (self.energy_time() - initial_energy) /  initial_energy)
        plt.xlabel(r'$t$', fontsize=14)
        plt.ylabel(r'$\frac{\Delta E(t)}{E(0)}$', fontsize=14)
