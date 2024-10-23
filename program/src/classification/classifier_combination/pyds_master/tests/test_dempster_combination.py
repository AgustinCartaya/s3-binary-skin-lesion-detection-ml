import numpy as np
import cv2 as cv
from functions import  functions as fc

from pyds_master.pyds import MassFunction
# http://bennycheung.github.io/dempster-shafer-theory-for-classification


def extract_angles(images, pixel, normalize=True, max_value=1):
    arr = np.zeros((len(images)))

    counter = 0
    for image in images:
        arr[counter] = image[pixel[0], pixel[1]]
        counter+=1
    if normalize:
        arr /= max_value
    return arr

def angles2prob(angles, th):
    probs = np.zeros((angles.shape[0], 2))
    # background
    if np.all(angles >= 0):

        # center engles in th
        centered_angles = angles-th
        for i in range(angles.shape[0]):
            # closer to tumor
            if centered_angles[i] >= 0:
                probs[i, 0] = 0.5 + centered_angles[i] / (2*(1-th))
                probs[i, 1] = 1 - probs[i, 0]
            else:
                probs[i, 1] = 0.5 + (-1) * centered_angles[i] / (2*th)
                probs[i, 0] = 1 - probs[i, 1]

        return probs
    
    return None

def weigthed_mass(probabilities, weights, y):
    masses = np.zeros((probabilities.shape))

    for i in range(len(probabilities)):
        if probabilities[i,0] > probabilities[i,1]:
            masses[i,0] = min(0.99, probabilities[i,0] + probabilities[i,0] * weights[i] * y)
            masses[i,1] = 1 - masses[i,0]
        elif probabilities[i,1] > probabilities[i,0]:
            masses[i,1] = min(0.99, probabilities[i,1] + probabilities[i,1] * weights[i] * y)
            masses[i,0] = 1 - masses[i,1]
        else:
            masses[i, :] = probabilities[i,:]
    return masses


def naif_mass(weights, probabilities):
    masses = np.zeros((probabilities.shape))

    masses[:,0] = weights * probabilities[:,0]
    masses[:,1] = weights * probabilities[:,1]
    # print(np.sum(masses))
    # masses /= np.sum(masses)
    return masses


def compute_denoeux_mass(weights, probabilities, y = 1):
    masses = np.zeros((probabilities.shape))

    masses[:,0] = weights * np.exp(-y*(probabilities[:,0]**2))
    masses[:,1] = weights * np.exp(-y*(probabilities[:,1]**2))
    return masses


def compute_appriou_mass(weights, probabilities, y = 1):
    masses = np.zeros((probabilities.shape))
    masses[:,0] = weights * (1-y*probabilities[:,0])
    masses[:,1] = weights * (1-y*probabilities[:,1])
    return masses

def dempster_combination(masses):
    img_res = np.zeros((masses.shape[0], masses.shape[1]))

    for pix_y in range(masses.shape[0]):
        for pix_x in range(masses.shape[1]):
            mass_funtions = []
            total_mass = np.sum(masses[pix_y, pix_x, :])
            if total_mass > 0:
                for mass in masses[pix_y, pix_x]:
                
                    mass_funtions.append(MassFunction({"M": mass[0], "S": mass[1]}))
                    # mass_funtions.append(MassFunction({"M": mass[1]/total_mass, "S": mass[0]/total_mass}))

                combined_mass_function = mass_funtions[0].combine_conjunctive(mass_funtions[1:])
                img_res[pix_y, pix_x] = combined_mass_function["M"]
                print(combined_mass_function)
            else:
                img_res[pix_y, pix_x] = -1
    return img_res

def simple_dempster_combination(masses):
    combined_prob = 0
    mass_funtions = []
    for mass in masses:
        mass_funtions.append(MassFunction({"M": mass[0], "S": mass[1]}))
    combined_mass_function = mass_funtions[0].combine_conjunctive(mass_funtions[1:])
    combined_prob = combined_mass_function["M"]
    return combined_prob


def test():
    sources_shape = (2,2)
    sources = []
    sources_weights = [1, 1]

    source1 = np.zeros((sources_shape[0],sources_shape[1])) + 1
    source2 = np.zeros((sources_shape[0],sources_shape[1])) + 0.3
    # source3 = np.zeros((sources_shape[0],sources_shape[1])) + 0.1
    # source4 = np.zeros((sources_shape[0],sources_shape[1])) + 0.1

    sources.append(source1)
    sources.append(source2)
    # sources.append(source3)
    # sources.append(source4)

    masses = np.zeros((sources_shape[0],sources_shape[1], len(sources), 2))
    img_res = np.zeros(sources_shape)
    for pix_y in range(sources_shape[0]):
        for pix_x in range(sources_shape[1]):
            angles_array = extract_angles(sources,pixel=(pix_y, pix_x), normalize=True, max_value=1)
            probs = angles2prob(angles_array, 0.5)
            # masses[pix_y, pix_x] = probs
            # print(probs)

            if probs is not None:
            #     # masses[pix_y, pix_x] = compute_denoeux_mass(sources_weights, probs, y=0.5)
            #     # masses[pix_y, pix_x] = compute_appriou_mass(sources_weights, probs, y=1)
            #     masses[pix_y, pix_x] = naif_mass(sources_weights, probs)
                masses = weigthed_mass(probs, sources_weights, 0.5)
                print("probs:")
                print(probs)
                print("masses:")
                print(masses)


                img_res[pix_y, pix_x] = simple_dempster_combination(masses)





    # img_res = dempster_combination(masses)
    print("mean", np.mean(img_res))
    print("std", np.std(img_res))

    fc.imgshow( "img combination" , img_res, scale=120)

    source_counter = 1
    for source in sources:
        fc.imgshow( "source" + str(source_counter) , source, scale=120)
        source_counter +=1

test()




def test_array():
    masses = np.array([0.50075618, 0.50803923 ,1, 0.51603112, 0, 0.47272384, 0, 0.54193321])
    res = dempster_combination( masses)
    print(res)

test_array()

cv.waitKey(0)
cv.destroyAllWindows()