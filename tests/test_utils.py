import unittest
import numpy as np
import scipy

import mdprop

class TestUtils(unittest.TestCase):
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
        
    def test_symbol_to_mass(self):
        self.assertTrue(self.M[0] == 1.007825032 * mdprop.units.AMU_TO_AU)
        self.assertTrue(self.M[2] == 15.99491462 * mdprop.units.AMU_TO_AU)

    def test_density_to_spherical_radius(self):
        density = 1.0 # g/ml
        r = mdprop.utils.density_to_spherical_radius(self.M, density)
        self.assertAlmostEqual(r, 3.6388275220920807)

    def test_compute_center_of_mass(self):
        com = (self.X[0] * self.M[0] + self.X[1] * self.M[1] + self.X[2] * self.M[2]) / np.sum(self.M)
        self.assertTrue(np.array_equal(com, mdprop.utils.compute_center_of_mass(self.X, self.M)))

    def test_compute_angular_momentum(self):
        tam = np.zeros((3,))
        for i in range(3):
            tam += np.cross(self.X[i], self.V[i]) * self.M[i]
        self.assertTrue(np.array_equal(tam, mdprop.utils.compute_angular_momentum(self.X, self.V, self.M)))

    def test_compute_kinetic_energy(self):
        ke = 0.0
        for i in range(3):
            ke += 0.5 * np.sum(self.M[i] * self.V[i] * self.V[i])
        mdp_ke = mdprop.utils.compute_kinetic_energy(self.V, self.M)
        self.assertAlmostEqual(ke, mdp_ke)

    def test_compute_temperature(self):
        t = 0.0
        for i in range(3):
            t += np.sum(self.M[i] * self.V[i] * self.V[i])
        t *= 1.0/float(np.size(self.X))
        mdp_t = mdprop.utils.compute_temperature(self.V, self.M)
        self.assertAlmostEqual(t, mdp_t)

    def test_pairwise_dist(self):
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.X))
        self.assertTrue(np.array_equal(dist, mdprop.utils.pairwise_dist(self.X)))

    def test_symbol_to_covalent_radius(self):
        r = mdprop.utils.symbol_to_covalent_radius(self.S)
        self.assertTrue(r[0] == 0.31 * mdprop.units.ANGSTROM_TO_AU)
        self.assertTrue(r[2] == 0.66 * mdprop.units.ANGSTROM_TO_AU)
                    
    def test_covalent_dist_matrix(self):
        d = mdprop.utils.covalent_dist_matrix(self.S)
        self.assertAlmostEqual(d[0, 0], 0.31*2 * mdprop.units.ANGSTROM_TO_AU)
        self.assertAlmostEqual(d[1, 0], 0.31*2 * mdprop.units.ANGSTROM_TO_AU)
        self.assertAlmostEqual(d[2, 0], 0.31 * mdprop.units.ANGSTROM_TO_AU + 0.66 * mdprop.units.ANGSTROM_TO_AU)
        self.assertAlmostEqual(d[2, 2], 0.66*2 * mdprop.units.ANGSTROM_TO_AU)

    def test_align(self):
        Rgen = mdprop.utils.random_rotation_matrix()
        Xrot = np.dot(self.X, Rgen)
        Xrot_cent = Xrot - mdprop.utils.compute_center_of_mass(Xrot, self.M)
        Xcent = self.X - mdprop.utils.compute_center_of_mass(self.X, self.M)
        Xrec, R = mdprop.utils.align(Xrot_cent, Xcent, self.M)
        Rabs = np.abs(R)
        Rgen_abs = np.abs(Rgen) # Need absolute values (and transpose) due to weirdness in phases of rotation matrix
        self.assertTrue(np.allclose(Xrec, Xcent))
        self.assertTrue(np.allclose(Rabs, Rgen_abs) or np.allclose(Rabs, Rgen_abs.T))
        
    def test_numerical_gradient(self):
        def func(x):
            return x[0] * x[0] + 3.0 * x[0] * x[1]
        def grad(x):
            return np.array([2.0 * x[0] + 3.0 * x[1], 3.0 * x[0]])
        xval = np.array([4.5, 2.5])
        g = grad(xval)
        num_g = mdprop.utils.numerical_gradient(xval, func)
        self.assertTrue(np.allclose(g, num_g))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)
