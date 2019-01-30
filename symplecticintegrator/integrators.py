import numpy as np
from symplecticintegrator.base import Integrator

__all__ = ['SymplecticIntegrator', 'RungeKutta4Integrator']


#region Symplectic integrators

class PositionVerletIntegrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        q_mid = q_curr + 1./2 * p_curr * dt
        p_next = p_curr + self.dp_dt(q_mid, p_curr) * dt
        q_next = q_mid + 1./2 * p_next * dt
        return q_next, p_next


class VelocityVerletIntegrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        p_mid = p_curr + 1./2 * self.dp_dt(q_curr, p_curr) * dt
        q_next = q_curr + p_mid * dt
        p_next = p_mid + 1./2 * self.dp_dt(q_next, p_curr) * dt
        return q_next, p_next


class SymplecticIntegrator(Integrator):
    nth_order_coefficients = {
        1: ([1], [1]),
        2: ([0, 1], [0.5, 0.5]),
        3: ([1, -2./3, 2./3], [-1./24, 3./4, 7./24]),
        4: ([1. / (2 * (2 - 2**(1./3))), (1 - 2**(1./3)) / (2 * (2 - 2**(1./3))),
             (1 - 2**(1./3)) / (2 * (2 - 2**(1./3))), 1. / (2 * (2 - 2**(1./3)))],
            [1. / (2 - 2**(1./3)), -2**(1./3) / (2 - 2**(1./3)), 1. / (2 - 2**(1./3)), 0])
    }

    def _create_single_step_func(self, order):
        c_list, d_list = self.nth_order_coefficients.get(order, 1)

        def single_step_func(q_curr, p_curr, dt):
            q_next = np.copy(q_curr)
            p_next = np.copy(p_curr)
            for c, d in zip(c_list, d_list):
                q_curr = q_next
                p_curr = p_next
                q_next = q_curr + dt * c * self.dq_dt(q_curr, p_curr)
                p_next = p_curr + dt * d * self.dp_dt(q_next, p_curr)
            return q_next, p_next

        return single_step_func

    def __init__(self, hamiltonian, q_symbols, p_symbols, parameter_subs=None, order=1):
        super().__init__(hamiltonian, q_symbols, p_symbols, parameter_subs)
        self.order = order
        self.single_step_func = self._create_single_step_func(order)

#endregion


#region Non-Symplectic integrators

class NothingIntegrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        return q_curr, p_curr


class EulerIntegrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        q_next = q_curr + dt * self.dq_dt(q_curr, p_curr)
        p_next = p_curr + dt * self.dp_dt(q_curr, p_curr)
        return q_next, p_next


class MidpointIntegrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        q_mid = q_curr + 1./2 * self.dq_dt(q_curr, p_curr) * dt
        p_mid = p_curr + 1./2 * self.dp_dt(q_curr, p_curr) * dt
        q_next = q_curr + self.dq_dt(q_mid, p_mid) * dt
        p_next = p_curr + self.dp_dt(q_mid, p_mid) * dt
        return q_next, p_next


class RungeKutta4Integrator(Integrator):
    def single_step_func(self, q_curr, p_curr, dt):
        k1_q = self.dq_dt(q_curr, p_curr)
        k1_p = self.dp_dt(q_curr, p_curr)

        k2_q = self.dq_dt(q_curr + 1./2 * k1_q * dt, p_curr + 1./2 * k1_p * dt)
        k2_p = self.dp_dt(q_curr + 1./2 * k1_q * dt, p_curr + 1./2 * k1_p * dt)

        k3_q = self.dq_dt(q_curr + 1./2 * k2_q * dt, p_curr + 1./2 * k2_p * dt)
        k3_p = self.dp_dt(q_curr + 1./2 * k2_q * dt, p_curr + 1./2 * k2_p * dt)

        k4_q = self.dq_dt(q_curr + k3_q * dt, p_curr + k3_p * dt)
        k4_p = self.dp_dt(q_curr + k3_q * dt, p_curr + k3_p * dt)

        q_next = q_curr + 1./6 * (k1_q + 2*k2_q + 2*k3_q + k4_q) * dt
        p_next = p_curr + 1./6 * (k1_p + 2*k2_p + 2*k3_p + k4_p) * dt
        return q_next, p_next

#endregion
