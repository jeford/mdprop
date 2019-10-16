import numpy as np

def exponential(state, beta):
    """ Returns G(y) where
    Q(q) = \\exp(-beta U(q))
    Q(p) = 0
    G(y) = - \\nabla U^T M^{-1} p 
    Note: the desired gradient must be added via a hook to the state
    """
    return -beta * np.sum(state['gradient'] * state['V'])
    
def balanced(state, beta):
    """ Returns G(y) where
    Q(q) = \\exp(-beta U(q))
    Q(p) = \\exp(-beta p^T M^{-1} p)
    Q(y) = Q(p) + Q(q)
    G(y) = - beta * \\frac{Q(q) - Q(p)}{Q(q) + Q(p)} \\nabla U^T M^{-1} p
    Note: the desired gradient must be added via a hook to the state
    """
    Qp = np.exp(-beta * state['kinetic_energy'])
    Qq = np.exp(-beta * state['potential_energy'])
    Kdot = -beta * np.sum(state['gradient'] * state['V'])
    return Kdot * (Qq - Qp) / (Qq + Qp)

def component(state, beta):
    """ Returns G(y) where
    Q(p)_i = \exp(-\\beta/2 ||p_i||^2 m_i^{-1})
    Q(p) = \sum_i Q(p)_i
    G(y) = \\frac{-\sum_{ij} \\beta p_{ij} m_i^{-1} \\nabla U_{ij} Q(p)_i}{Q(p)}
    Note: the desired gradient must be added via a hook to the state

    This control function should penalize any one mode/atom from gaining too 
    much kinetic energy as in an isokinetic ensemble
    """
    Qps = np.exp(beta * 0.5 * np.sum(state['V']**2, axis=1, keepdims=True) * np.reshape(state['masses'], (-1, 1)))
    return - np.sum(beta * state['V'] * state['gradient'] * Qps) / np.sum(Qps)
