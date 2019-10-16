import unittest
import numpy as np

import mdprop

class TestData(unittest.TestCase):
    def test_elements(self):
        self.assertTrue(mdprop.data.element_dict['Xx'].symbol == 'Xx')
        self.assertTrue(mdprop.data.element_dict['Xx'].name == 'Dummy')
        self.assertTrue(mdprop.data.element_dict['Xx'].atomic_num == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].exact_mass == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].covalent_radius == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].vdw_radius == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].bond_radius == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].electronegativity == 0.0)
        self.assertTrue(mdprop.data.element_dict['Xx'].max_bonds == 0.0)
        
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestData)
    unittest.TextTestRunner(verbosity=2).run(suite)
