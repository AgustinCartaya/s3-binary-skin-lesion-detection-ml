
import sys
import os

# add functions code 
ruta_del_paquete = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_del_paquete)

import numpy as np
from .pyds_master.pyds import MassFunction


def dempster_combination(masses):
    combined_prob = 0
    mass_funtions = []
    for mass in masses:
        mass_funtions.append(MassFunction({"M": mass, "S": 1 - mass}))
    # mass_funtions.reverse()
    combined_mass_function = mass_funtions[0].combine_conjunctive(mass_funtions[1:])
    combined_prob = combined_mass_function["M"]
    return combined_prob


def dempster_combination_multiclass(masses):
    """colums represent labels, rows represents beleaves"""
    nb_classes = masses[0].size
    combined_probs = np.zeros((nb_classes))
    mass_funtions = []
    for mass in masses:
        comb_dict = {}
        for i in range(nb_classes):
            comb_dict[str(i)] = mass[i]
        mass_funtions.append(MassFunction(comb_dict))

    combined_mass_function = mass_funtions[0].combine_conjunctive(mass_funtions[1:])

    for i in range(nb_classes):
        combined_probs[i] = combined_mass_function[str(i)]

    return combined_probs


def min_distance_combination(masses):
    combined_prob = 0
    min_index = -1
    min_value = -1
    counter = 0
    for mass in masses:
        if mass > 0.5:
            _min_value = 1 - mass
        else:
            _min_value = mass

        if min_index == -1 or _min_value < min_value:
            min_index = counter
        counter =+1
    combined_prob = masses[min_index]
    return combined_prob


def mean_combination(masses):
    return np.mean(masses, axis=1)


def mean_combination_multiclass(masses):
    """colums represent labels, rows represents beleaves"""
    return np.mean(masses, axis=1)
