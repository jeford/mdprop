"""All element data through the whole period table.

This module was part of the Nanoreactor package.

Data are loaded from OpenBabel using the following script:

    import pybel as pb
    from collections import namedtuple

    Element = namedtuple("Element", ["symbol", "name", "atomic_num", "exact_mass",
                                     "covalent_radius", "vdw_radius", "bond_radius",
                                     "electronegativity", "max_bonds"])
    element_data = []
    elements = pb.ob.OBElementTable()
    isotopes = pb.ob.OBIsotopeTable()
    for i in xrange(elements.GetNumberOfElements()):
        new_element = Element(elements.GetSymbol(i), elements.GetName(i), i,
                              isotopes.GetExactMass(i), elements.GetCovalentRad(i),
                              elements.GetVdwRad(i), elements.CorrectedBondRad(i),
                              elements.GetElectroNeg(i), elements.GetMaxBonds(i))
        element_data.append(new_element)
    print repr(element_data)

"""

from collections import namedtuple, OrderedDict

Element = namedtuple("Element", ["symbol", "name", "atomic_num", "exact_mass",
                                 "covalent_radius", "vdw_radius", "bond_radius",
                                 "electronegativity", "max_bonds"])

element_list = [
    Element(symbol='Xx', name='Dummy', atomic_num=0, exact_mass=0.0,
              covalent_radius=0.0, vdw_radius=0.0, bond_radius=0.0,
              electronegativity=0.0, max_bonds=0),
    Element(symbol='H', name='Hydrogen', atomic_num=1, exact_mass=1.007825032,
              covalent_radius=0.31, vdw_radius=1.1, bond_radius=0.31,
              electronegativity=2.2, max_bonds=1),
    Element(symbol='He', name='Helium', atomic_num=2, exact_mass=4.002603254,
              covalent_radius=0.28, vdw_radius=1.4, bond_radius=0.28,
              electronegativity=0.0, max_bonds=0),
    Element(symbol='Li', name='Lithium', atomic_num=3, exact_mass=7.01600455,
              covalent_radius=1.28, vdw_radius=1.81, bond_radius=1.28,
              electronegativity=0.98, max_bonds=1),
    Element(symbol='Be', name='Beryllium', atomic_num=4, exact_mass=9.0121822,
              covalent_radius=0.96, vdw_radius=1.53, bond_radius=0.96,
              electronegativity=1.57, max_bonds=2),
    Element(symbol='B', name='Boron', atomic_num=5, exact_mass=11.0093054,
              covalent_radius=0.84, vdw_radius=1.92, bond_radius=0.84,
              electronegativity=2.04, max_bonds=4),
    Element(symbol='C', name='Carbon', atomic_num=6, exact_mass=12.0,
              covalent_radius=0.76, vdw_radius=1.7, bond_radius=0.76,
              electronegativity=2.55, max_bonds=4),
    Element(symbol='N', name='Nitrogen', atomic_num=7, exact_mass=14.003074005,
              covalent_radius=0.71, vdw_radius=1.55, bond_radius=0.71,
              electronegativity=3.04, max_bonds=4),
    Element(symbol='O', name='Oxygen', atomic_num=8, exact_mass=15.99491462,
              covalent_radius=0.66, vdw_radius=1.52, bond_radius=0.66,
              electronegativity=3.44, max_bonds=2),
    Element(symbol='F', name='Fluorine', atomic_num=9, exact_mass=18.99840322,
              covalent_radius=0.57, vdw_radius=1.47, bond_radius=0.57,
              electronegativity=3.98, max_bonds=1),
    Element(symbol='Ne', name='Neon', atomic_num=10, exact_mass=19.992440175,
              covalent_radius=0.58, vdw_radius=1.54, bond_radius=0.58,
              electronegativity=0.0, max_bonds=0),
    Element(symbol='Na', name='Sodium', atomic_num=11, exact_mass=22.989769281,
              covalent_radius=1.66, vdw_radius=2.27, bond_radius=1.66,
              electronegativity=0.93, max_bonds=1),
    Element(symbol='Mg', name='Magnesium', atomic_num=12, exact_mass=23.9850417,
              covalent_radius=1.41, vdw_radius=1.73, bond_radius=1.41,
              electronegativity=1.31, max_bonds=2),
    Element(symbol='Al', name='Aluminium', atomic_num=13, exact_mass=26.98153863,
              covalent_radius=1.21, vdw_radius=1.84, bond_radius=1.21,
              electronegativity=1.61, max_bonds=6),
    Element(symbol='Si', name='Silicon', atomic_num=14, exact_mass=27.976926532,
              covalent_radius=1.11, vdw_radius=2.1, bond_radius=1.11,
              electronegativity=1.9, max_bonds=6),
    Element(symbol='P', name='Phosphorus', atomic_num=15, exact_mass=30.97376163,
              covalent_radius=1.07, vdw_radius=1.8, bond_radius=1.07,
              electronegativity=2.19, max_bonds=6),
    Element(symbol='S', name='Sulfur', atomic_num=16, exact_mass=31.972071,
              covalent_radius=1.05, vdw_radius=1.8, bond_radius=1.05,
              electronegativity=2.58, max_bonds=6),
    Element(symbol='Cl', name='Chlorine', atomic_num=17, exact_mass=34.96885268,
              covalent_radius=1.02, vdw_radius=1.75, bond_radius=1.02,
              electronegativity=3.16, max_bonds=1),
    Element(symbol='Ar', name='Argon', atomic_num=18, exact_mass=39.962383123,
              covalent_radius=1.06, vdw_radius=1.88, bond_radius=1.06,
              electronegativity=0.0, max_bonds=0),
    Element(symbol='K', name='Potassium', atomic_num=19, exact_mass=38.96370668,
              covalent_radius=2.03, vdw_radius=2.75, bond_radius=2.03,
              electronegativity=0.82, max_bonds=1),
    Element(symbol='Ca', name='Calcium', atomic_num=20, exact_mass=39.96259098,
              covalent_radius=1.76, vdw_radius=2.31, bond_radius=1.76,
              electronegativity=1.0, max_bonds=2),
    Element(symbol='Sc', name='Scandium', atomic_num=21, exact_mass=44.9559119,
              covalent_radius=1.7, vdw_radius=2.3, bond_radius=1.7,
              electronegativity=1.36, max_bonds=6),
    Element(symbol='Ti', name='Titanium', atomic_num=22, exact_mass=47.9479463,
              covalent_radius=1.6, vdw_radius=2.15, bond_radius=1.6,
              electronegativity=1.54, max_bonds=6),
    Element(symbol='V', name='Vanadium', atomic_num=23, exact_mass=50.9439595,
              covalent_radius=1.53, vdw_radius=2.05, bond_radius=1.53,
              electronegativity=1.63, max_bonds=6),
    Element(symbol='Cr', name='Chromium', atomic_num=24, exact_mass=51.9405075,
              covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
              electronegativity=1.66, max_bonds=6),
    Element(symbol='Mn', name='Manganese', atomic_num=25, exact_mass=54.9380451,
              covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
              electronegativity=1.55, max_bonds=8),
    Element(symbol='Fe', name='Iron', atomic_num=26, exact_mass=55.9349375,
              covalent_radius=1.32, vdw_radius=2.05, bond_radius=1.32,
              electronegativity=1.83, max_bonds=6),
    Element(symbol='Co', name='Cobalt', atomic_num=27, exact_mass=58.933195,
              covalent_radius=1.26, vdw_radius=2.0, bond_radius=1.26,
              electronegativity=1.88, max_bonds=6),
    Element(symbol='Ni', name='Nickel', atomic_num=28, exact_mass=57.9353429,
              covalent_radius=1.24, vdw_radius=2.0, bond_radius=1.24,
              electronegativity=1.91, max_bonds=6),
    Element(symbol='Cu', name='Copper', atomic_num=29, exact_mass=62.9295975,
              covalent_radius=1.32, vdw_radius=2.0, bond_radius=1.32,
              electronegativity=1.9, max_bonds=6),
    Element(symbol='Zn', name='Zinc', atomic_num=30, exact_mass=63.929142,
              covalent_radius=1.22, vdw_radius=2.1, bond_radius=1.22,
              electronegativity=1.65, max_bonds=6),
    Element(symbol='Ga', name='Gallium', atomic_num=31, exact_mass=68.925573,
              covalent_radius=1.22, vdw_radius=1.87, bond_radius=1.22,
              electronegativity=1.81, max_bonds=3),
    Element(symbol='Ge', name='Germanium', atomic_num=32, exact_mass=73.921177,
              covalent_radius=1.2, vdw_radius=2.11, bond_radius=1.2,
              electronegativity=2.01, max_bonds=4),
    Element(symbol='As', name='Arsenic', atomic_num=33, exact_mass=74.921596,
              covalent_radius=1.19, vdw_radius=1.85, bond_radius=1.19,
              electronegativity=2.18, max_bonds=3),
    Element(symbol='Se', name='Selenium', atomic_num=34, exact_mass=79.916521,
              covalent_radius=1.2, vdw_radius=1.9, bond_radius=1.2,
              electronegativity=2.55, max_bonds=2),
    Element(symbol='Br', name='Bromine', atomic_num=35, exact_mass=78.918337,
              covalent_radius=1.2, vdw_radius=1.83, bond_radius=1.2,
              electronegativity=2.96, max_bonds=1),
    Element(symbol='Kr', name='Krypton', atomic_num=36, exact_mass=83.911507,
              covalent_radius=1.16, vdw_radius=2.02, bond_radius=1.16,
              electronegativity=3.0, max_bonds=0),
    Element(symbol='Rb', name='Rubidium', atomic_num=37, exact_mass=84.911789,
              covalent_radius=2.2, vdw_radius=3.03, bond_radius=2.2,
              electronegativity=0.82, max_bonds=1),
    Element(symbol='Sr', name='Strontium', atomic_num=38, exact_mass=87.905612,
              covalent_radius=1.95, vdw_radius=2.49, bond_radius=1.95,
              electronegativity=0.95, max_bonds=2),
    Element(symbol='Y', name='Yttrium', atomic_num=39, exact_mass=88.905848,
              covalent_radius=1.9, vdw_radius=2.4, bond_radius=1.9,
              electronegativity=1.22, max_bonds=6),
    Element(symbol='Zr', name='Zirconium', atomic_num=40, exact_mass=89.904704,
              covalent_radius=1.75, vdw_radius=2.3, bond_radius=1.75,
              electronegativity=1.33, max_bonds=6),
    Element(symbol='Nb', name='Niobium', atomic_num=41, exact_mass=92.906378,
              covalent_radius=1.64, vdw_radius=2.15, bond_radius=1.64,
              electronegativity=1.6, max_bonds=6),
    Element(symbol='Mo', name='Molybdenum', atomic_num=42, exact_mass=97.905408,
              covalent_radius=1.54, vdw_radius=2.1, bond_radius=1.54,
              electronegativity=2.16, max_bonds=6),
    Element(symbol='Tc', name='Technetium', atomic_num=43, exact_mass=97.907216,
              covalent_radius=1.47, vdw_radius=2.05, bond_radius=1.47,
              electronegativity=1.9, max_bonds=6),
    Element(symbol='Ru', name='Ruthenium', atomic_num=44, exact_mass=101.904349,
              covalent_radius=1.46, vdw_radius=2.05, bond_radius=1.46,
              electronegativity=2.2, max_bonds=6),
    Element(symbol='Rh', name='Rhodium', atomic_num=45, exact_mass=102.905504,
              covalent_radius=1.42, vdw_radius=2.0, bond_radius=1.42,
              electronegativity=2.28, max_bonds=6),
    Element(symbol='Pd', name='Palladium', atomic_num=46, exact_mass=105.903486,
              covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
              electronegativity=2.2, max_bonds=6),
    Element(symbol='Ag', name='Silver', atomic_num=47, exact_mass=106.905097,
              covalent_radius=1.45, vdw_radius=2.1, bond_radius=1.45,
              electronegativity=1.93, max_bonds=6),
    Element(symbol='Cd', name='Cadmium', atomic_num=48, exact_mass=113.903358,
              covalent_radius=1.44, vdw_radius=2.2, bond_radius=1.44,
              electronegativity=1.69, max_bonds=6),
    Element(symbol='In', name='Indium', atomic_num=49, exact_mass=114.903878,
              covalent_radius=1.42, vdw_radius=2.2, bond_radius=1.42,
              electronegativity=1.78, max_bonds=3),
    Element(symbol='Sn', name='Tin', atomic_num=50, exact_mass=119.902194,
              covalent_radius=1.39, vdw_radius=1.93, bond_radius=1.39,
              electronegativity=1.96, max_bonds=4),
    Element(symbol='Sb', name='Antimony', atomic_num=51, exact_mass=120.903815,
              covalent_radius=1.39, vdw_radius=2.17, bond_radius=1.39,
              electronegativity=2.05, max_bonds=3),
    Element(symbol='Te', name='Tellurium', atomic_num=52, exact_mass=129.906224,
              covalent_radius=1.38, vdw_radius=2.06, bond_radius=1.38,
              electronegativity=2.1, max_bonds=2),
    Element(symbol='I', name='Iodine', atomic_num=53, exact_mass=126.904473,
              covalent_radius=1.39, vdw_radius=1.98, bond_radius=1.39,
              electronegativity=2.66, max_bonds=1),
    Element(symbol='Xe', name='Xenon', atomic_num=54, exact_mass=131.904153,
              covalent_radius=1.4, vdw_radius=2.16, bond_radius=1.4,
              electronegativity=2.6, max_bonds=0),
    Element(symbol='Cs', name='Caesium', atomic_num=55, exact_mass=132.905451,
              covalent_radius=2.44, vdw_radius=3.43, bond_radius=2.44,
              electronegativity=0.79, max_bonds=1),
    Element(symbol='Ba', name='Barium', atomic_num=56, exact_mass=137.905247,
              covalent_radius=2.15, vdw_radius=2.68, bond_radius=2.15,
              electronegativity=0.89, max_bonds=2),
    Element(symbol='La', name='Lanthanum', atomic_num=57, exact_mass=138.906353,
              covalent_radius=2.07, vdw_radius=2.5, bond_radius=2.07,
              electronegativity=1.1, max_bonds=12),
    Element(symbol='Ce', name='Cerium', atomic_num=58, exact_mass=139.905438,
              covalent_radius=2.04, vdw_radius=2.48, bond_radius=2.04,
              electronegativity=1.12, max_bonds=6),
    Element(symbol='Pr', name='Praseodymium', atomic_num=59, exact_mass=140.907652,
              covalent_radius=2.03, vdw_radius=2.47, bond_radius=2.03,
              electronegativity=1.13, max_bonds=6),
    Element(symbol='Nd', name='Neodymium', atomic_num=60, exact_mass=141.907723,
              covalent_radius=2.01, vdw_radius=2.45, bond_radius=2.01,
              electronegativity=1.14, max_bonds=6),
    Element(symbol='Pm', name='Promethium', atomic_num=61, exact_mass=144.912749,
              covalent_radius=1.99, vdw_radius=2.43, bond_radius=1.99,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Sm', name='Samarium', atomic_num=62, exact_mass=151.919732,
              covalent_radius=1.98, vdw_radius=2.42, bond_radius=1.98,
              electronegativity=1.17, max_bonds=6),
    Element(symbol='Eu', name='Europium', atomic_num=63, exact_mass=152.92123,
              covalent_radius=1.98, vdw_radius=2.4, bond_radius=1.98,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Gd', name='Gadolinium', atomic_num=64, exact_mass=157.924103,
              covalent_radius=1.96, vdw_radius=2.38, bond_radius=1.96,
              electronegativity=1.2, max_bonds=6),
    Element(symbol='Tb', name='Terbium', atomic_num=65, exact_mass=158.925346,
              covalent_radius=1.94, vdw_radius=2.37, bond_radius=1.94,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Dy', name='Dysprosium', atomic_num=66, exact_mass=163.929174,
              covalent_radius=1.92, vdw_radius=2.35, bond_radius=1.92,
              electronegativity=1.22, max_bonds=6),
    Element(symbol='Ho', name='Holmium', atomic_num=67, exact_mass=164.930322,
              covalent_radius=1.92, vdw_radius=2.33, bond_radius=1.92,
              electronegativity=1.23, max_bonds=6),
    Element(symbol='Er', name='Erbium', atomic_num=68, exact_mass=165.930293,
              covalent_radius=1.89, vdw_radius=2.32, bond_radius=1.89,
              electronegativity=1.24, max_bonds=6),
    Element(symbol='Tm', name='Thulium', atomic_num=69, exact_mass=168.934213,
              covalent_radius=1.9, vdw_radius=2.3, bond_radius=1.9,
              electronegativity=1.25, max_bonds=6),
    Element(symbol='Yb', name='Ytterbium', atomic_num=70, exact_mass=173.938862,
              covalent_radius=1.87, vdw_radius=2.28, bond_radius=1.87,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Lu', name='Lutetium', atomic_num=71, exact_mass=174.940771,
              covalent_radius=1.87, vdw_radius=2.27, bond_radius=1.87,
              electronegativity=1.27, max_bonds=6),
    Element(symbol='Hf', name='Hafnium', atomic_num=72, exact_mass=179.94655,
              covalent_radius=1.75, vdw_radius=2.25, bond_radius=1.75,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Ta', name='Tantalum', atomic_num=73, exact_mass=180.947995,
              covalent_radius=1.7, vdw_radius=2.2, bond_radius=1.7,
              electronegativity=1.5, max_bonds=6),
    Element(symbol='W', name='Tungsten', atomic_num=74, exact_mass=183.950931,
              covalent_radius=1.62, vdw_radius=2.1, bond_radius=1.62,
              electronegativity=2.36, max_bonds=6),
    Element(symbol='Re', name='Rhenium', atomic_num=75, exact_mass=186.955753,
              covalent_radius=1.51, vdw_radius=2.05, bond_radius=1.51,
              electronegativity=1.9, max_bonds=6),
    Element(symbol='Os', name='Osmium', atomic_num=76, exact_mass=191.96148,
              covalent_radius=1.44, vdw_radius=2.0, bond_radius=1.44,
              electronegativity=2.2, max_bonds=6),
    Element(symbol='Ir', name='Iridium', atomic_num=77, exact_mass=192.962926,
              covalent_radius=1.41, vdw_radius=2.0, bond_radius=1.41,
              electronegativity=2.2, max_bonds=6),
    Element(symbol='Pt', name='Platinum', atomic_num=78, exact_mass=194.964791,
              covalent_radius=1.36, vdw_radius=2.05, bond_radius=1.36,
              electronegativity=2.28, max_bonds=6),
    Element(symbol='Au', name='Gold', atomic_num=79, exact_mass=196.966568,
              covalent_radius=1.36, vdw_radius=2.1, bond_radius=1.36,
              electronegativity=2.54, max_bonds=6),
    Element(symbol='Hg', name='Mercury', atomic_num=80, exact_mass=201.970643,
              covalent_radius=1.32, vdw_radius=2.05, bond_radius=1.32,
              electronegativity=2.0, max_bonds=6),
    Element(symbol='Tl', name='Thallium', atomic_num=81, exact_mass=204.974427,
              covalent_radius=1.45, vdw_radius=1.96, bond_radius=1.45,
              electronegativity=1.62, max_bonds=3),
    Element(symbol='Pb', name='Lead', atomic_num=82, exact_mass=207.976652,
              covalent_radius=1.46, vdw_radius=2.02, bond_radius=1.46,
              electronegativity=2.33, max_bonds=4),
    Element(symbol='Bi', name='Bismuth', atomic_num=83, exact_mass=208.980398,
              covalent_radius=1.48, vdw_radius=2.07, bond_radius=1.48,
              electronegativity=2.02, max_bonds=3),
    Element(symbol='Po', name='Polonium', atomic_num=84, exact_mass=208.98243,
              covalent_radius=1.4, vdw_radius=1.97, bond_radius=1.4,
              electronegativity=2.0, max_bonds=2),
    Element(symbol='At', name='Astatine', atomic_num=85, exact_mass=209.987148,
              covalent_radius=1.5, vdw_radius=2.02, bond_radius=1.5,
              electronegativity=2.2, max_bonds=1),
    Element(symbol='Rn', name='Radon', atomic_num=86, exact_mass=222.017577,
              covalent_radius=1.5, vdw_radius=2.2, bond_radius=1.5,
              electronegativity=0.0, max_bonds=0),
    Element(symbol='Fr', name='Francium', atomic_num=87, exact_mass=223.019735,
              covalent_radius=2.6, vdw_radius=3.48, bond_radius=2.6,
              electronegativity=0.7, max_bonds=1),
    Element(symbol='Ra', name='Radium', atomic_num=88, exact_mass=226.025409,
              covalent_radius=2.21, vdw_radius=2.83, bond_radius=2.21,
              electronegativity=0.9, max_bonds=2),
    Element(symbol='Ac', name='Actinium', atomic_num=89, exact_mass=227.027752,
              covalent_radius=2.15, vdw_radius=2.0, bond_radius=2.15,
              electronegativity=1.1, max_bonds=6),
    Element(symbol='Th', name='Thorium', atomic_num=90, exact_mass=232.038055,
              covalent_radius=2.06, vdw_radius=2.4, bond_radius=2.06,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Pa', name='Protactinium', atomic_num=91, exact_mass=231.035884,
              covalent_radius=2.0, vdw_radius=2.0, bond_radius=2.0,
              electronegativity=1.5, max_bonds=6),
    Element(symbol='U', name='Uranium', atomic_num=92, exact_mass=238.050788,
              covalent_radius=1.96, vdw_radius=2.3, bond_radius=1.96,
              electronegativity=1.38, max_bonds=6),
    Element(symbol='Np', name='Neptunium', atomic_num=93, exact_mass=237.048173,
              covalent_radius=1.9, vdw_radius=2.0, bond_radius=1.9,
              electronegativity=1.36, max_bonds=6),
    Element(symbol='Pu', name='Plutonium', atomic_num=94, exact_mass=244.064204,
              covalent_radius=1.87, vdw_radius=2.0, bond_radius=1.87,
              electronegativity=1.28, max_bonds=6),
    Element(symbol='Am', name='Americium', atomic_num=95, exact_mass=243.061381,
              covalent_radius=1.8, vdw_radius=2.0, bond_radius=1.8,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Cm', name='Curium', atomic_num=96, exact_mass=247.070354,
              covalent_radius=1.69, vdw_radius=2.0, bond_radius=1.69,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Bk', name='Berkelium', atomic_num=97, exact_mass=247.070307,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Cf', name='Californium', atomic_num=98, exact_mass=251.079587,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Es', name='Einsteinium', atomic_num=99, exact_mass=252.08298,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Fm', name='Fermium', atomic_num=100, exact_mass=257.095105,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Md', name='Mendelevium', atomic_num=101, exact_mass=258.098431,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='No', name='Nobelium', atomic_num=102, exact_mass=259.10103,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=1.3, max_bonds=6),
    Element(symbol='Lr', name='Lawrencium', atomic_num=103, exact_mass=262.10963,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Rf', name='Rutherfordium', atomic_num=104, exact_mass=261.10877,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Db', name='Dubnium', atomic_num=105, exact_mass=262.11408,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Sg', name='Seaborgium', atomic_num=106, exact_mass=263.11832,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Bh', name='Bohrium', atomic_num=107, exact_mass=264.1246,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Hs', name='Hassium', atomic_num=108, exact_mass=265.13009,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Mt', name='Meitnerium', atomic_num=109, exact_mass=268.13873,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Ds', name='Darmstadtium', atomic_num=110, exact_mass=281.162061,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Rg', name='Roentgenium', atomic_num=111, exact_mass=280.164473,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Cn', name='Copernicium', atomic_num=112, exact_mass=285.174105,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Uut', name='Ununtrium', atomic_num=113, exact_mass=284.17808,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Fl', name='Flerovium', atomic_num=114, exact_mass=289.187279,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Uup', name='Ununpentium', atomic_num=115, exact_mass=288.192492,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Lv', name='Livermorium', atomic_num=116, exact_mass=292.199786,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Uus', name='Ununseptium', atomic_num=117, exact_mass=292.20755,
              covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
              electronegativity=0.0, max_bonds=6),
    Element(symbol='Uuo', name='Ununoctium', atomic_num=118, exact_mass=293.21467,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6)
]

element_dict = OrderedDict([(e.symbol, e) for e in element_list])

