import unittest
import numpy as np

import mdprop

class TestInit(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
                    [ 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [ 0.0, 1.0, 0.0],
                    ])
        self.V = np.array([
                    [ 0.0, 0.0, 0.0],
                    [ 0.0, 0.0, 0.0],
                    [ 1.0, 0.0, 0.0],
                    ])
        self.S = ['H', 'H', 'O']
        self.M = mdprop.utils.symbol_to_mass(self.S)
        
    def test_center(self):
        self.assertFalse(np.allclose(mdprop.utils.compute_center_of_mass(self.X, self.M), np.zeros(3,)))
        self.assertFalse(np.allclose(mdprop.utils.compute_center_of_mass(self.V, self.M), np.zeros(3,)))
        Xnew = mdprop.init.center(self.X, self.M)
        Vnew = mdprop.init.center(self.V, self.M)
        self.assertTrue(np.allclose(mdprop.utils.compute_center_of_mass(Xnew, self.M), np.zeros(3,)))
        self.assertTrue(np.allclose(mdprop.utils.compute_center_of_mass(Vnew, self.M), np.zeros(3,)))

    def test_angular_momentum_components(self):
        dV = mdprop.init.angular_momentum_components(self.X, self.V, self.M)
        X = self.X
        V = self.V + dV
        tam = mdprop.utils.compute_angular_momentum(X, V, self.M)
        self.assertTrue(np.allclose(tam, np.zeros((3, ))))

    def test_initialize_centered(self):
        self.assertFalse(np.allclose(mdprop.utils.compute_angular_momentum(self.X, self.V, self.M), np.zeros(3,)))
        X, V = mdprop.init.initialize_centered(self.X, self.V, self.M)
        com = mdprop.utils.compute_center_of_mass(X, self.M)
        comV = mdprop.utils.compute_center_of_mass(V, self.M)
        tam = mdprop.utils.compute_angular_momentum(X, V, self.M)
        self.assertTrue(np.allclose(com, np.zeros((3, ))))
        self.assertTrue(np.allclose(comV, np.zeros((3, ))))
        self.assertTrue(np.allclose(tam, np.zeros((3, ))))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInit)
    unittest.TextTestRunner(verbosity=2).run(suite)
