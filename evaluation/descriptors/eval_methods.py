import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict
import cv2
import eval_support_functions as esf

"""Evaluationsfunktionen für Deskriptoren."""

def eval_num_max_possible_matches(kpts_i:np.array, kpts_j:np.array) -> int:
    return np.min([kpts_i.shape[0], kpts_j.shape[0]])

def eval_num_found_matches(matches: List) -> int:
    return len(matches)

def eval_perc_found_matches(num_max_possible_matches: int, num_found_matches:int) -> float:
    return num_found_matches / num_max_possible_matches

def eval_num_matches_ratio_test(matches: List) -> int:
    return len(matches)

def eval_perc_matches_ratio_test(num_matches:int, num_matches_ratio_test) -> float:
    return num_matches_ratio_test / num_matches

def eval_num_inlier_fmatrix(inlier_mask:np.array) -> int:
    if inlier_mask is None:
        return 0
    else:
        return np.sum(inlier_mask)

def eval_perc_inlier_fmatrix(num_inlier:int, num_matches:int) -> float:
    return num_inlier / num_matches

def eval_perc_inlier_ratio_test_fmatrix(num_inlier:int, num_matches_ratio_test:int) -> float:
    if num_inlier == 0:
        return 0
        
    return num_inlier / num_matches_ratio_test

def eval_sum_dist_epilines(distances) -> float:
    return np.sum(distances)

def eval_mean_dist_epilines(distances) -> float:
    return np.mean(distances)

def eval_std_dist_epilines(distances) -> float:
    return np.std(distances)

def eval_image_pairs(kpts_files:List[str], desc_files:List[str], config:Dict) -> Dict:
    num_files = len(kpts_files)
    flann = esf.get_flann()

    # Get all names for image_pair evaluations
    # Create for each name a pandas Dataframe to store information for each image pair.
    dfs_names = [x[17:] for x in list(config.keys()) if x.startswith('eval_image_pair__') and config[x]]
    dfs = dict((x, pd.DataFrame(data=np.zeros((num_files, num_files)))) for x in  dfs_names)
    
    for i in range(num_files):
        kpts_i = pd.read_csv(kpts_files[i], sep=',', comment='#', names=['x', 'y', 'size', 'angle', 'response', 'octave', 'class_id'])
        desc_i = pd.read_csv(desc_files[i], sep=',', comment='#').values.astype('float32') 

        for j in range(num_files):
            kpts_j = pd.read_csv(kpts_files[j], sep=',', comment='#', names=['x', 'y', 'size', 'angle', 'response', 'octave', 'class_id'])
            desc_j = pd.read_csv(desc_files[j], sep=',', comment='#').values.astype('float32')

            num_max_possible_matches = eval_num_max_possible_matches(kpts_i, kpts_j)

            if config['eval_image_pair__num_max_possible_matches']:
                dfs['num_max_possible_matches'].iloc[i, j] = num_max_possible_matches

            matches = flann.knnMatch(desc_i, desc_j, k=2)

            num_found_matches = eval_num_found_matches(matches)

            if config['eval_image_pair__num_found_matches']:
                dfs['num_found_matches'].iloc[i, j] = num_found_matches
            
            if config['eval_image_pair__perc_found_matches']:
                dfs['perc_found_matches'].iloc[i, j] = eval_perc_found_matches(num_max_possible_matches, num_found_matches)

            good, pts_i, pts_j, match_ids = esf.apply_ratio_test_to_matches(matches, kpts_i, kpts_j)

            num_matches_ratio_test = eval_num_matches_ratio_test(good)

            if config['eval_image_pair__num_matches_ratio_test']:
                dfs['num_matches_ratio_test'].iloc[i, j] = num_matches_ratio_test

            if config['eval_image_pair__perc_matches_ratio_test']:
                dfs['perc_matches_ratio_test'].iloc[i, j] = eval_perc_matches_ratio_test(num_found_matches, num_matches_ratio_test)

            F, mask = esf.compute_fundamental_matrix(pts_i, pts_j)

            num_inlier_fmatrix = eval_num_inlier_fmatrix(mask)

            if config['eval_image_pair__num_inlier_fmatrix']:
                dfs['num_inlier_fmatrix'].iloc[i, j] = num_inlier_fmatrix
            
            if config['eval_image_pair__perc_inlier_fmatrix']:
                dfs['perc_inlier_fmatrix'].iloc[i, j] =  eval_perc_inlier_fmatrix(num_inlier_fmatrix, num_found_matches)
            
            if config['eval_image_pair__perc_inlier_ratio_test_fmatrix']:
                dfs['perc_inlier_ratio_test_fmatrix'].iloc[i, j] =  eval_perc_inlier_ratio_test_fmatrix(num_inlier_fmatrix, num_matches_ratio_test)

            inliers_i = esf.get_inliers_from_mask(pts_i, mask)
            inliers_j = esf.get_inliers_from_mask(pts_j, mask)

            distances = esf.compute_distances_kpts_to_epilines(inliers_i, inliers_j, F)

            if config['eval_image_pair__sum_dist_epilines_first_image']:
                dfs['sum_dist_epilines_first_image'].iloc[i, j] = eval_sum_dist_epilines(distances[:, 0])
            
            if config['eval_image_pair__sum_dist_epilines_second_image']:
                dfs['sum_dist_epilines_second_image'].iloc[i, j] = eval_sum_dist_epilines(distances[:, 1])
            
            if config['eval_image_pair__sum_dist_epilines']:
                dfs['sum_dist_epilines'].iloc[i, j] = eval_sum_dist_epilines(distances)

            if config['eval_image_pair__mean_dist_epilines_first_image']:
                dfs['mean_dist_epilines_first_image'].iloc[i, j] = eval_mean_dist_epilines(distances[:, 0])
            
            if config['eval_image_pair__mean_dist_epilines_second_image']:
                dfs['mean_dist_epilines_second_image'].iloc[i, j] = eval_mean_dist_epilines(distances[:, 1])
            
            if config['eval_image_pair__mean_dist_epilines']:
                dfs['mean_dist_epilines'].iloc[i, j] = eval_mean_dist_epilines(distances)

            if config['eval_image_pair__std_dist_epilines_first_image']:
                dfs['std_dist_epilines_first_image'].iloc[i, j] = eval_std_dist_epilines(distances[:, 0])
            
            if config['eval_image_pair__std_dist_epilines_second_image']:
                dfs['std_dist_epilines_second_image'].iloc[i, j] = eval_std_dist_epilines(distances[:, 1])
            
            if config['eval_image_pair__std_dist_epilines']:
                dfs['std_dist_epilines'].iloc[i, j] = eval_std_dist_epilines(distances)

    return dfs















