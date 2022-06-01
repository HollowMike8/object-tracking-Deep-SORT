import numpy as np
import scipy.linalg
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

'''
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
'''
chi2inv95 = {1: 3.8415,
             2: 5.9915,
             3: 7.8147,
             4: 9.4877,
             5: 11.070,
             6: 12.592,
             7: 14.067,
             8: 15.507,
             9: 16.919}

class KalmanBoxTracker(object):
    '''
    This class represents state of individual tracked objects observed as bbox.
    '''
    
    def __init__(self):
        '''
        Intialize the kalman tracker with the given bounding box
            
        Returns
        -------
        None.
        
        '''
        self.dim_x, self.dim_z, self.dt = 8, 4, 1
    
    def initiate(self, measurement):
        '''
        Create track from unassociated measurement.
        
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
            
        Returns
        -------
        self.kf.x : ndarray
            The 8 dimensional mean vector of the object state at the given
            time step.
        self.kf.P : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            given time step.
        '''
        
        # use constant velocity model (A*X + w)
        # X:[x, vx, y, vy, a, va, h, vh] and Z:[x, y, a, h]
        
        self.kf = KalmanFilter(self.dim_x, self.dim_z)
        
        # state transition matrix
        self.kf.F = np.array([[1, self.dt, 0,       0, 0,       0, 0,       0],
                              [0,       1, 0,       0, 0,       0, 0,       0],
                              [0,       0, 1, self.dt, 0,       0, 0,       0],
                              [0,       0, 0,       1, 0,       0, 0,       0],
                              [0,       0, 0,       0, 1, self.dt, 0,       0],
                              [0,       0, 0,       0, 0,       1, 0,       0],
                              [0,       0, 0,       0, 0,       0, 1, self.dt],
                              [0,       0, 0,       0, 0,       0, 0,       1]])

        # process noise matrix (discrete)--> assumption:x,y,a,h are independent
        q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.001)
        self.kf.Q = block_diag(q, q, q, q)
        
        # control matrix --> default = None
        self.kf.B = None
        
        # measurement function --> Z = H*X
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])

        # measurement noise matrix --> variance of x, y, a, h in pixel^2
        self.kf.R = np.array([[10, 0, 0, 0],
                              [0, 10, 0, 0],
                              [0, 0, 10, 0],
                              [0, 0, 0, 10]])
        
        # initial conditions --> covariance matrix
        self.kf.P = np.eye(8)*500
        
        # initial conditions --> object state vector
        self.kf.x = self._convert_xyah_to_x(measurement)
        
        return self.kf.x, self.kf.P

    def predict(self):
        '''
        Run Kalman tracker prediction step.
        
        Returns
        -------
        self.kf.x : ndarray
            The 8 dimensional mean vector of the object state at the given
            time step.
        self.kf.P : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            given time step.
        '''

        self.kf.predict()
        
        return self.kf.x, self.kf.P

    def update(self, measurement):
        '''
        Run Kalman tracker update step.

        Parameters
        ----------
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        
        Returns
        -------
        self.kf.x : ndarray
            The 8 dimensional mean vector of the object state at the given
            time step.
        self.kf.P : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            given time step.
        '''
        
        if measurement.size != 0:
            z = np.dot(self.kf.H, self._convert_xyah_to_x(measurement))
            self.kf.update(z)

        return self.kf.x, self.kf.P

    def project(self):
        '''
        Project state vector to measurement space.

        Returns
        -------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the given
            time step.
        covariance + innovation_cov : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            given time step.
        '''
        mean = self._convert_x_to_xyah(self.kf.x)
        
        innovation_cov = self.kf.S
        covariance = np.array([[self.kf.P[0][0], 0, 0, 0],
                               [0, self.kf.P[2][2], 0, 0],
                               [0, 0, self.kf.P[4][4], 0],
                               [0, 0, 0, self.kf.P[6][6]]])
        
        return mean, covariance + innovation_cov

    def gating_distance(self, measurements, only_position=False):
        '''
        Compute mahalanobis distance between state vector and measurements.If
        `only_position` is False, the computation uses 4 degrees of freedom 
        ([x, y, a, h]), otherwise 2 ([x, y]).
        
        Parameters
        ----------
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
            
        Returns
        -------
        squared_maha : ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between state vector and measurements.
            
        '''
        mean, covariance = self.project()
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        
        return squared_maha          
        
    def _convert_xyah_to_x(self, xyah):
        '''
        Converts bbox parameters (x, y, a, h) into 
        [x, vx, y, vy, a, va, h, vh] where x,y is the centre of the bbox, 
        a is the aspect ratio (`width / height`) and h is the height and their
        respective velocities.
        
        NOTE: This function is useful only for initialization and update as we 
        don't deal with actual velocities
        '''
        x = [xyah[int(i/2)] if i%2==0 else 0 for i in range(2*len(xyah))]
        
        return np.asarray(x).reshape((8,1))

    def _convert_x_to_xyah(self, x):
        '''
        Takes state vector and returns it in the form [x,y,a,h] where x,y is the 
        centre of bbox and a is aspect ratio (`width / height`) and h is height
        '''
        xyah = [x[int(i)] for i in range(len(x)) if i%2==0]
        
        return np.asarray(xyah).reshape((1,4))