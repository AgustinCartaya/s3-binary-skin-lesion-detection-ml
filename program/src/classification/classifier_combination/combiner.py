import numpy as np 

from . import mass_creator as mass_creator
from . import  combination_methods as combination_methods

MASS_SIGMOID = "sigmoid_mass"
MASS_VECTOR = "vector_mass"
MASS_LINEAR = "linear_mass"

COMBINATION_DEMPSTER = "dempster_combination"
COMBINATION_MIN_DISTANCE = "min_distance_combination"
COMBINATION_MEAN = "mean_combination"


def combine_results(scores, weights, mass_method, combination_method, weights_influence=0):
    shape_scores = scores.shape
    # conver values to mass
    masses = np.zeros_like(scores)
    for i in range(shape_scores[0]):
        if mass_method is None:
            masses = scores.copy()
        if mass_method == MASS_SIGMOID:
            masses[i] = mass_creator.sigmoid_mass(scores[i], th=0.5, weights=np.array([weights[i],0,0]), weights_influence=weights_influence)
            # masses[i] = mass_creator.sigmoid_mass(scores[i], th=0.5, weights=weights[i], weights_influence=0)
        elif mass_method == MASS_VECTOR:
            masses[i] = mass_creator.vector_mass(scores[i], th=0.5, weights=np.array([weights[i],0,0]), weights_influence=weights_influence)
        elif mass_method == MASS_LINEAR:
            masses[i] = mass_creator.linear_mass(scores[i], th=0.5, weights=np.array([weights[i], weights[i], weights[i]]), weights_influence=weights_influence)
            

    # combine values
    combined_scores = np.zeros((shape_scores[1]))
    for i in range(shape_scores[1]):
        if combination_method == COMBINATION_DEMPSTER:
            combined_scores[i] = combination_methods.dempster_combination(masses[:,i])
        elif combination_method == COMBINATION_MEAN:
            combined_scores[i] = combination_methods.mean_combination(masses[:,i])
        elif combination_method == COMBINATION_MIN_DISTANCE:
            combined_scores[i] = combination_methods.min_distance_combination(masses[:,i])

    return combined_scores



def combine_results_multiclass(scores_list):
    combined_data = np.zeros_like(scores_list[0])
    for i in range(scores_list[0].shape[0]):

        _to_combine_list = []
        for cscore in scores_list:
            _to_combine_list.append(cscore[i])

        _to_combine = np.vstack(_to_combine_list)
        combined_data[i] = combination_methods.dempster_combination_multiclass(_to_combine)

    return combined_data