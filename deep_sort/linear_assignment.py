from __future__ import absolute_import

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import kalman_filter

INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections, 
                      track_indices=None, detection_indices=None):
    '''
    Solve linear assignment problem (cost minimization problem).

    Parameters
    ----------
    distance_metric : ndarray
        Callable[List[Track], List[Detection], List[int], List[int])              
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating/distance threshold. Associations with cost larger than this 
        value are disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step 
        (defined from track.py).
    detections : List[detection.Detection]
        A list of detections at the current time step
        (defined from detection.py).
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks.
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections.

    Returns
    -------
    matches : List[(int, int)]
    unmatched_tracks : List[int]
    unmatched_detections : List[int]

    '''
    
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
        
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    
    # apply minimization problem on cost matrix and extract matching indices
    cost_matrix = distance_metric(tracks, detections, track_indices, 
                                  detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_sum_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    
    # initiate empty lists for matches, unmatched_tracks, unmatched_detections
    matches, unmatched_tracks, unmatched_detections = [], [], []
    
    for col,detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
            
    for row,track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    
    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, 
                     detections, track_indices=None, detection_indices=None):
    '''
    Run matching cascade.

    Parameters
    ----------
    distance_metric : ndarray
        Callable[List[Track], List[Detection], List[int], List[int])              
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating/distance threshold. Associations with cost larger than this 
        value are disregarded.
    cascade_depth : int
        Maximum age of the track.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step 
        (defined from track.py).
    detections : List[detection.Detection]
        A list of detections at the current time step
        (defined from detection.py).
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks.
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections.

    Returns
    -------
    matches : List[(int, int)]
    unmatched_tracks : List[int]
    unmatched_detections : List[int]

    '''
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    unmatched_detections = detection_indices
    matches = []
    
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        
        track_indices_l = [k for k in track_indices 
                           if tracks[k].time_since_update == 1 + level]
        
        if len(track_indices_l) == 0:
            # check for matching at next level
            continue
        
        # capture the matchings and update the unmatched_detections 
        matches_l, _, unmatched_detections = \
            min_cost_matching(distance_metric, max_distance, tracks, 
                              detections, track_indices_l, unmatched_detections)
        matches += matches_l
    
    # capture the unmatched_tracks
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, 
                     detection_indices, gated_cost=INFTY_COST, 
                     only_position=False):
    '''
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : List of Kalman filters.
    cost_matrix : ndarray
        DESCRIPTION.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step 
        (defined from track.py).
    detections : List[detection.Detection]
        A list of detections at the current time step
        (defined from detection.py).
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks.
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections.
    gated_cost : float, optional
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. The default is INFTY_COST.
    only_position : bool, optional
        If True, only the x, y position of the state distribution is considered
        during gating. The default is False.

    Returns
    -------
    cost_matrix : ndarray
        The modified cost matrix

    '''
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf[track.track_id-1].gating_distance(measurements, 
                                                               only_position)  
        
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        
    return cost_matrix    