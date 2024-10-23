import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.integrate import nquad

def bhattacharyya_distance(hist1, hist2):
    # Calculate the measure of Bhattacharyya
    bhatt_coeff = np.sum(np.sqrt(np.multiply(hist1, hist2)))
    
    # Calculate distance from Bhattacharyya
    distance = max(0,-np.log(bhatt_coeff))
    
    return distance

def hellinger_distance(hist1, hist2):
    # Calculate the Hellinger distance
    distance = np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2))**2)) / np.sqrt(2)
    
    return distance

def hellinger_distance_multivariable(distribution_P, distribution_Q):
    # Función para calcular la distancia de Hellinger
    def integrand(*args):
        p = distribution_P(*args)
        q = distribution_Q(*args)
        return 0.5 * (np.sqrt(p) - np.sqrt(q))**2

    # Calcula la integral sobre todo el espacio de variables
    result, _ = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])
    distance = np.sqrt(result)

    return distance

def mahalanobis_distance(hist_a, hist_b):
    """"
        compute mahalanobis distance betweeng two vectors
        args:
        hist_a: vector a
        hist_b: vector b
    """
    hist_a = hist_a.reshape((1, 15))
    hist_b = hist_b.reshape((1, 15))

    cov_hists = np.cov(np.vstack((hist_a, hist_b)), rowvar=False)

    regularization = 0.01 
    cov_hists += regularization * np.identity(cov_hists.shape[0])

    mahalanobis_dist = mahalanobis(hist_a[0,:], hist_b[0,:], np.linalg.inv(cov_hists))

    return mahalanobis_dist

def patch_mahalanobis_distance(pixels, pixels_ref):
    """"
        compute mahalanobis distance from each pixel to a distribution
        args:
        pixels: contains all pixels to compute the distance
        pixels_ref: contains all the pixels representing the distribution
    """
    mu = np.mean(pixels_ref)
    std = np.std(pixels_ref)

    dist = np.sum(np.sqrt(((pixels-mu)**2)/std)) / pixels.shape[0]
    return dist


def pearson_correlation(hist1, hist2):
    # Calculate the Pearson correlation coefficient
    correlation = np.corrcoef(hist1, hist2)[0, 1]

    # Normalize the correlation coefficient to the range [0, 1]
    similarity = (correlation + 1) / 2

    return 1-similarity


def jensen_shannon_distance(hist1, hist2):
    # Calculate the average histogram
    avg_hist = 0.5 * (hist1 + hist2)

    # Calculate the Kullback-Leibler divergences between histograms and the average histogram
    data1 = np.log2(hist1 / avg_hist)
    data_cleaned1 = np.where(np.isinf(data1) & (data1 < 0), 0, data1)
    data_cleaned1 = np.nan_to_num(data_cleaned1)

    data2 = np.log2(hist2 / avg_hist)
    data_cleaned2 = np.where(np.isinf(data2) & (data2 < 0), 0, data2)
    data_cleaned2 = np.nan_to_num(data_cleaned2)

    kl_div_1 = np.sum(hist1 * data_cleaned1)
    kl_div_2 = np.sum(hist2 * data_cleaned2)

    # Calculate the Jensen-Shannon distance as half the sum of Kullback-Leibler divergences
    jensen_shannon_dist = 0.5 * (kl_div_1 + kl_div_2)
    return jensen_shannon_dist


def kl_divergence_entropy(hist1, hist2):
    kl_divergence = entropy(hist1, hist2)
    return kl_divergence

def kl_divergence_unilateral(p, q):
    # Evitar divisiones por cero y valores negativos
    p[p == 0] = 1e-10
    q[q == 0] = 1e-10

    # Calcular la divergencia de Kullback-Leibler
    kl = np.sum(p * np.log(p / q))
    
    return kl

def kl_divergence_bilateral(hist1, hist2):
    kl1 = kl_divergence_unilateral(hist1, hist2)
    kl2 = kl_divergence_unilateral(hist2, hist1)
    return kl1 + kl2


def kl_divergence_normalized(hist1, hist2):
    def __kl_divergence_normalized(s1, s2):
        # normalize
        k1 = np.sum(s1)
        k2 = np.sum(s2)

        # Evitar divisiones por cero y valores negativos
        s1[s1 == 0] = 1e-10
        s2[s2 == 0] = 1e-10

        # Calcular la divergencia de Kullback-Leibler
        kld = np.sum(s1 * np.log(s1 / s2))/k1 + np.log(k2/k1)
    
        return kld, k1
    

    kl1, k1 = __kl_divergence_normalized(hist1, hist2)
    kl2, k2 = __kl_divergence_normalized(hist2, hist1)
    return k1 * kl1 + k2 * kl2 + (k1-k2) * np.log(k1/k2)


def patch_kl_divergence(patch1, patch2):
    patch1 = 5000 + patch1
    patch2 = 5000 + patch2

    return kl_divergence_normalized(patch1, patch2)

def earth_mover_distance(hist1, hist2):
    distance = wasserstein_distance(hist1, hist2)
    return distance

def combined_earth_mover_distance(H1, H2, alpha=0.1, beta=10):
    EMD = wasserstein_distance(H1, H2)
    norm_diff = np.linalg.norm(H1) - np.linalg.norm(H2)
    DNEMD = (1/alpha) * EMD + (1/beta) * norm_diff
    return DNEMD

# def ECS_distance(s1, s2):
#     sum_totla = 0
#     for lmd in range(1, s1.shape[0]):
#         sum_s1 = np.sum(s1[:lmd])
#         sum_s2 = np.sum(s2[:lmd])
#         sum_totla += (sum_s1 - sum_s2)**2
#     return np.sqrt(sum_totla)  

def ECS_distance(s1, s2):
    cumulative_s1 = np.cumsum(s1)  # Calculate the cumulative sum of s1
    cumulative_s2 = np.cumsum(s2)  # Calculate the cumulative sum of s2
    squared_difference = (cumulative_s1 - cumulative_s2) ** 2  # Calculate the squared differences
    sum_squared_difference = np.sum(squared_difference)  # Sum all the squared differences
    return np.sqrt(sum_squared_difference)  # Returns the square root of the sum


distance_name_function = {
    "bhattacharyya": bhattacharyya_distance,
    "hellinger": hellinger_distance,
    "mahalanobis": mahalanobis_distance,
    # "patch_mahalanobis": patch_mahalanobis_distance,
    "pearson_correlation": pearson_correlation,
    "jensen_shannon": jensen_shannon_distance,
    "kl_divergence_entropy": kl_divergence_entropy,
    "kl_divergence_unilateral": kl_divergence_unilateral,
    "kl_divergence_bilateral": kl_divergence_bilateral,
    "kl_divergence_normalized": kl_divergence_normalized,
    # "patch_kl_divergence": patch_kl_divergence,
    "earth_mover": earth_mover_distance,
    "combined_EMD": combined_earth_mover_distance,
    "ECS": ECS_distance,
}

def get_distance_name_function(name):
    return (name, distance_name_function[name])
