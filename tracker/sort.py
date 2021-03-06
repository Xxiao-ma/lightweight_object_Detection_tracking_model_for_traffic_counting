"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from kalman_tracker import KalmanBoxTracker
#from correlation_tracker import CorrelationTracker
from data_association import associate_detections_to_trackers


class Sort:
  # reach best result with min_hits=3
  def __init__(self,max_age=9,min_hits=3, use_dlib = False):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    # the format looks like: [det num,trk id]
    self.det_trk_pair = []


    self.use_dlib = use_dlib

  def update(self,dets,img=None):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1

    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    matched = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict(img) 
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    if dets != []:
      umatched_trks = None
      unmatched_dets = None
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
     
      #update matched trackers with assigned detections
      for t,trk in enumerate(self.trackers):

        if(t not in unmatched_trks):
          matched_p = matched[np.where(matched[:,1]==t)]
          d = matched[np.where(matched[:,1]==t)[0],0]
          trk.update(dets[d,:][0],img) 
          self.det_trk_pair.append([d,trk.id])
        else:
          # update it self by its own prediction if the tracker is unmatched,
          # this will preserve the tracker, so that it can be used potentially in the case of rematching after occlusion.
          if dets != []:
            trk.update(trk.predict(),img)

      #create and initialise new trackers for unmatched detections
      for i in unmatched_dets:
        if not self.use_dlib:
          trk = KalmanBoxTracker(dets[i,:])
          self.det_trk_pair.append([i,trk.id])
        else:
          print("error")
        self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        if dets == []:
          trk.update([],img)
        d = trk.get_state()

        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive

        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    for trk in self.trackers:
        p  = trk.get_gt()
        ids = trk.get_id() 
        self.det_trk_pair.append([p, ids])

    if(len(ret)>0):
      return np.concatenate(ret), matched
    return np.empty((0,5)),[]