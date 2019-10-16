import numpy as np
import pickle as pkl
import json

from . import units

def read_xyz(filename, unit_conversion=units.ANGSTROM_TO_AU):
    """
    Read xyz file of atomic coordinates.

    Args:
        filename: xyz file to read coordinates from.
    """
    X = []
    symbols = []
    with open(filename, 'r') as fin:
        lines = fin.readlines()
        natom = int(lines[0])
        for l in lines[2:2+natom]:
            sp = l.split()
            symbols.append(sp[0])
            X.append([float(s) for s in sp[1:]])
            #X.append((float(sp[1]), float(sp[2]), float(sp[3])))
    X = np.array(X) * unit_conversion
    return X, np.array(symbols)

def read_traj_xyz(filename, unit_conversion=units.ANGSTROM_TO_AU, spaced=False):
    """
    Read xyz, vel trajectory file.

    Args:
        filename: xyz file to read coordinates from.
        unit_conversion: multiplicative factor to change units of file
        spaced: True if there is a space between frames, otherwise false

    Returns:
        X, symbols: list of frames of coordinates, and list of atomic symbols
    """
    X = []
    currX = []
    symbols = []
    natom = 0
    frame = -1
    space = int(spaced)

    # Extract data about all frames
    with open(filename, 'r') as fin:
        natom = int(fin.readline())
        fin.readline() # skip second line
        for i in range(natom):
            symbols.append(fin.readline().split()[0])

    # Extract coordinates
    with open(filename, 'r') as fin:
        for i, line in enumerate(fin):
            mod = i%(2+natom+space)
            if mod==0:
                if currX:
                    X.append(np.array(currX) * unit_conversion)
                currX = []
                frame += 1
            # We don't care about second lines of frame
            elif mod==1:
                pass
            # Read in lines
            elif mod>=2 and mod<2+natom:
                sp = line.split()
                currX.extend(list(map(float, sp[1:])))

    # Get final frame
    if currX:
        X.append(np.array(currX) * unit_conversion)
                
    return X, symbols

def read_xyz_bagel_input(filename, unit_conversion=1.0):
    """
    Read coordinates from BAGEL input json file.

    Args:
        filename: BAGEL input file to read coordinates from.
        unit_conversion: conversion factor of units, default 1.0 (a.u.)
    """
    with open(filename, 'r') as fin:
        dct = json.load(fin)
    S = [d['atom'] for d in dct['bagel'][0]['geometry']]
    X = np.array([d['xyz'] for d in dct['bagel'][0]['geometry']])
    return X, S

def read_xyz_bagel_output(filename, unit_conversion=1.0):
    """
    Read coordinates from BAGEL output file, returns all frames found

    Args:
        filename: BAGEL output file to read coordinates from.
        unit_conversion: conversion factor of units, default 1.0 (a.u.)

    Returns:
        Xs: list of (natom, 3) np.arrays of coordinates
        S: list of atomic symbols
    """
    Xs = []
    S = []
    ibegin = -2
    nframes = 0
    read = False
    with open(filename, 'r') as fin:
        for i, line in enumerate(fin):
            if "*** Geometry ***" in line:
                ibegin = i
                read = True
                nframes += 1
                Xcurr = []
            elif i == ibegin + 1:
                continue
            elif read and "atom" in line:
                sp = line.split()
                if nframes == 1:
                    S.append(sp[3][1:-2])
                Xcurr.append([sp[7][:-1], sp[8][:-1], sp[9][:-1]])
            elif read:
                Xs.append(np.array(Xcurr, dtype=np.float64))
                read = False
    return Xs, S

def save_xyz_filehandle(X, symbols, handle, text="", format_str="% .11E"):
    """
    Write coordinates and symbols to given file handle.

    Args:
        X: coordinates (can also append velocities or whatever)
        symbols: atomic symbols
        handle: file handle to write to
        text: Helper text in second line
        format_str: formatting for coordinates to write
    """
    Xsh = np.shape(X)
    handle.write("%d\n" % Xsh[0])
    handle.write(text)
    handle.write("\n")
    for i in range(Xsh[0]):
        handle.write(symbols[i])
        for j in range(Xsh[1]):
            handle.write("\t")
            handle.write(format_str % X[i, j])
        handle.write("\n")

def save_xyz(X, symbols, filename, unit_conversion=1.0/units.ANGSTROM_TO_AU, text="", format_str="% .11E"):
    """
    Write coordinates and symbols to given filename.

    Args:
        X: coordinates (can also append velocities or whatever)
        symbols: atomic symbols
        filename: file name to write to
        unit_conversion: unit conversion of coordinates (defaults to converting AU to Angstrom
        text: Helper text in second line
        format_str: formatting for coordinates to write
    """
    with open(filename, 'w') as fout:
        Xc = np.array(X) * unit_conversion
        save_xyz_filehandle(Xc, symbols, fout, text, format_str)

def save_xyz_bagel(X, symbols, filename, unit_conversion=1.0, format_str="% .11E"):
    """
    Write coordinates and symbols to given filename in BAGEL's json syntax in atomic units

    Args:
        X: coordinates (can also append velocities or whatever)
        symbols: atomic symbols
        filename: file name to write to
        unit_conversion: unit conversion of coordinates (defaults to converting AU to Angstrom
        text: Helper text in second line
        format_str: formatting for coordinates to write
    """
    # Construct dict
    Xc = np.array(X) * unit_conversion
    with open(filename, 'w') as fout:
        fout.write('\t{ "geometry" : [\n')
        for i, s in enumerate(symbols):
            fout.write(('\t\t{ "atom": "%s" , "xyz": [ ' + format_str + ', ' + format_str + ', ' + format_str + ' ] },\n') % (s, Xc[i, 0], Xc[i, 1], Xc[i, 2]))
        fout.write('\t] },')

def save_traj_xyz(Xs, symbols, filename, texts=None, unit_conversion=1.0/units.ANGSTROM_TO_AU, format_str="% .11E"):
    """
    Write coordinates and symbols to given filename for trajectory.

    Args:
        Xs: list of X coordinates to save
        symbols: atomic symbols
        filename: file name to write to
        texts: lines of text to print between atom number and symbols/coordinates
        unit_conversion: unit conversion of coordinates (defaults to converting AU to Angstrom
        format_str: formatting for coordinates to write
    """
    if texts is None:
        texts = ["Frame %d"%i for i in range(len(Xs))] 
    with open(filename, 'w') as fout:
        for i, X in enumerate(Xs):
            Xc = X * unit_conversion
            save_xyz_filehandle(Xc, symbols, fout, texts[i], format_str)

def save_traj_hist(history, filename, texts=None, unit_conversion=1.0/units.ANGSTROM_TO_AU, format_str="% .11E"):
    """
    Write coordinates and symbols to given filename for trajectory.

    Args:
        history: list of state dicts
        filename: file name to write to
        texts: lines of text to print between atom number and symbols/coordinates
        unit_conversion: unit conversion of coordinates (defaults to converting AU to Angstrom
        format_str: formatting for coordinates to write
    """
    if history[0].get('symbols', None) is None:
        raise ValueError("State key 'symbols' required to output xyz.")
    symbols = history[0]['symbols']
    if texts is None:
        texts = ["Frame %d, time %f" % (i, frame.get('simulation_time', 0.0)) for i, frame in enumerate(history)]
    Xs = [frame['X'] for frame in history]
    save_traj_xyz(Xs, symbols, filename, texts, unit_conversion, format_str)

def save_traj_hist_pkl(history, filename):
    with open(filename, 'wb') as fout:
        pkl.dump(history, fout, protocol=2)
        
def load_traj_hist_pkl(filename):
    with open(filename, 'rb') as fin:
        history = pkl.load(fin)
    return history
