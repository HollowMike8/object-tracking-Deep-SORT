import numpy as np

class Detection(object):
    '''
    Class representing the bounding box detection in a single image and 
    change bounding box format.

    Parameters
    ----------
    tlwh : ndarray
        Bounding box (bbox) in the format of (x, y, w, h) with top left as (x, y),
        width w, and height h.
    confidence : float
        Detector confidence score.
    class_name : string
        Class label of the detection in the bbox
    feature : ndarray
        A feature vector that describes the object contained in the bbox.
    '''
    
    def __init__(self, tlwh, confidence, class_name, feature):
        self.tlwh = np.asarray(tlwh, dtype = np.float)
        self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)
    
    def get_class(self):
        return self.class_name
    
    def to_tlbr(self):
        '''
        Converts input bbox format (x, y, w, h) to (min x, min y, max x, max y)

        Returns
        -------
        tlbr : ndarray
        Bounding box (bbox) in the format of (min x, min y, max x, max y).
        '''
        tlbr = self.tlwh.copy()
        tlbr[2:] += tlbr[:2]
        
        return tlbr
    
    def to_xyah(self):
        '''
        Converts input bbox format (x, y, w, h) to 
        (center x, center y, aspect ratio, height), 
        where the aspect ratio is `width / height`.

        Returns
        -------
        xyah : ndarray
        Bounding box in format of (center x, center y, aspect ratio, height).
        '''       
        xyah = self.tlwh.copy()
        xyah[:2] += xyah[2:] / 2
        xyah[2] /= xyah[3]
        
        return xyah