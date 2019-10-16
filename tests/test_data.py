import unittest
import numpy as np

import mdprop

class TestData(unittest.TestCase):
    def test_elements(self):
        self.assertTrue(mdprop.data.elements['Xx'].symbol == 'Xx')
        self.assertTrue(mdprop.data.elements['Xx'].name == 'Dummy')
        self.assertTrue(mdprop.data.elements['Xx'].atomic_num == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].exact_mass == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].covalent_radius == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].vdw_radius == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].bond_radius == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].electronegativity == 0.0)
        self.assertTrue(mdprop.data.elements['Xx'].max_bonds == 0.0)
        
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestData)
    unittest.TextTestRunner(verbosity=2).run(suite)
