
import numpy as np


# def sigmoid_mass_slow(img, th, weights, weights_influence):
#     centered_img = img-th
#     probs = np.zeros(centered_img.shape)
#     slope = weights[0]
#     for y in range(centered_img.shape[0]):
#         for x in range(centered_img.shape[1]):
#             for z in range(centered_img.shape[2]):
#                 if centered_img[y,x,z] > 0:
#                     desp = weights[2]
#                 elif centered_img[y,x,z] < 0:
#                     desp = -weights[1]
#                 else:
#                     desp = 0
#                 probs[y,x,z] = 1/(1+np.exp(-np.exp(weights_influence)* np.pi * slope * (centered_img[y,x,z]+desp)))
#     return probs


def sigmoid_mass(data, th, weights, weights_influence):
    centered_data = data - th
    positive_indices = centered_data > 0
    negative_indices = centered_data < 0

    desp = np.where(positive_indices, weights[2], np.where(negative_indices, -weights[1], 0))
    exponent = np.exp(-np.exp(weights_influence) * np.pi * weights[0] * (centered_data + desp))
    probs = 1 / (1 + exponent)
    
    return probs


# def vector_mass_slow(img, th, weights, weights_influence):
#     centered_img = img-th
#     probs = np.zeros(centered_img.shape)

#     for y in range(centered_img.shape[0]):
#         for x in range(centered_img.shape[1]):
#             for z in range(centered_img.shape[2]):
#                 if centered_img[y,x,z] > 0:
#                     probs[y,x,z] = 0.5 + centered_img[y,x,z] / (2*( (np.pi/2) - th))
#                     probs[y,x,z] = min(0.99, probs[y,x,z] + (probs[y,x,z] - 0.5) * weights[1] * weights_influence)
#                 elif centered_img[y,x,z] < 0:
#                     probs[y,x,z] = 0.5 + centered_img[y,x,z] / (2*th)
#                     probs[y,x,z] = max(0.01, probs[y,x,z] - (0.5 - probs[y,x,z]) * weights[2] * weights_influence)
#                 else:
#                     probs[y,x,z] = 0.5
#     return probs


def vector_mass(data, th, weights, weights_influence):
    centered_data = data - th
    probs = np.zeros(centered_data.shape)

    positive_indices = centered_data > 0
    negative_indices = centered_data < 0
    zero_indices = ~positive_indices & ~negative_indices

    probs[positive_indices] = 0.5 + centered_data[positive_indices] / (2 * ( 1 - th))
    probs[negative_indices] = 0.5 + centered_data[negative_indices] / (2 * th)
    probs[zero_indices] = 0.5

    probs[positive_indices] = np.minimum(0.99, probs[positive_indices] + (probs[positive_indices] - 0.5) * weights[1] * weights_influence)
    probs[negative_indices] = np.maximum(0.01, probs[negative_indices] - (0.5 - probs[negative_indices]) * weights[2] * weights_influence)

    return probs


# def linear_mass_slow(img, th, weights, weights_influence):
#     centered_img = img-th
#     probs = np.zeros(centered_img.shape)
#     slope_pos = np.tan(np.pi/2 * weights[2] * np.exp(weights_influence))
#     slope_neg = np.tan(np.pi/2 * weights[1] * np.exp(weights_influence))
#     for y in range(centered_img.shape[0]):
#         for x in range(centered_img.shape[1]):
#             for z in range(centered_img.shape[2]):
#                 if centered_img[y,x,z] >= 0:
#                     centered_img_normalized = (centered_img[y,x,z] / (np.pi/2 - th)) / 2
#                     probs[y,x,z] = 0.5 + min(0.49, slope_pos * centered_img_normalized)
#                 elif centered_img[y,x,z] < 0:
#                     centered_img_normalized = (centered_img[y,x,z] / th) / 2
#                     probs[y,x,z] = 0.5 + max(-0.49, slope_neg * centered_img_normalized)
#     return probs


def linear_mass(data, th, weights, weights_influence):
    centered_data = data - th
    centered_img_normalized_pos = (centered_data / (1- th)) / 2
    centered_img_normalized_neg = (centered_data / th) / 2
    
    slope_pos = np.tan(min(np.pi/2, np.pi/2 * weights[2] * np.exp(weights_influence)))
    slope_neg = np.tan(min(np.pi/2, np.pi/2 * weights[1] * np.exp(weights_influence)))

    pos_mask = centered_data >= 0
    neg_mask = ~pos_mask

    probs = np.zeros(centered_data.shape)

    probs[pos_mask] = 0.5 + np.minimum(0.49, slope_pos * centered_img_normalized_pos[pos_mask])
    probs[neg_mask] = 0.5 + np.maximum(-0.49, slope_neg * centered_img_normalized_neg[neg_mask])

    return probs

# img = np.array([[[-5, 0, 5],
#                 [-0.25, 0, 0.25],
#                 [-0.05, 0, 0.05]], [[-5, 0, 5],
#                                     [-0.25, 0, 0.25],
#                                     [-0.05, 0, 0.05]]])
# th = 0.1
# weights = [0.5, 0.51, 0.1]
# weights_influence = 0

# slow_mass = sigmoid_mass_slow(img, th=th, weights=weights, weights_influence=weights_influence)
# mass = sigmoid_mass(img, th=th, weights=weights, weights_influence=weights_influence)
# print(slow_mass, "\n\n", mass, "\n")
# print(np.array_equal(slow_mass, mass))

# ------------------------- 


# slow_mass = vector_mass_slow(img, th=th, weights=weights, weights_influence=weights_influence)
# mass = vector_mass(img, th=th, weights=weights, weights_influence=weights_influence)

# print(slow_mass, "\n\n", mass, "\n")
# print(np.array_equal(slow_mass, mass))

# # ------------------------- 

# slow_mass = linear_mass_slow(img, th=th, weights=weights, weights_influence=weights_influence)
# mass = linear_mass(img, th=th, weights=weights, weights_influence=weights_influence)

# print(slow_mass, "\n\n", mass, "\n")
# print(np.array_equal(slow_mass, mass))