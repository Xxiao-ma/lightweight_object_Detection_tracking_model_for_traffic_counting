import numpy as np
from filterpy.kalman import KalmanFilter
import pickle


'''Motion Model'''
class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,img=None):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    #defination of states x = [u,v,s,r,Δu,Δv,Δs]
    #where u and v represent the x and y location of target center, s and r represent scale(area) and aspect ratio
    #in total x has 7 dimentions
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.ground_truth = bbox

  def update(self,bbox,img=None):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    if bbox != []:
      self.kf.update(convert_bbox_to_z(bbox))

  def predict(self,img=None, dynamic_velocity=False, average_speed_ini= False):

    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    #use the generated model to predict velocity based on current position and object's size 

    #load the model
    if dynamic_velocity:
      with open("model_randomForest.pkl","rb") as f:
        model = pickle.load(f)

      # input format [x, w*h] for prediction with the given model
      w = np.sqrt(self.kf.x[2]*self.kf.x[3])
      h = self.kf.x[2]/w

      area = w*h
      direction = 1 # 1 means from left to right, while -1 means from right to left
      x_center = int(self.kf.x[0])
      
      #find out the velocity direction 
      if (int(self.kf.x[4])==0 and int(self.kf.x[0])>160) or int(self.kf.x[4])<0:
        x_center = 320-int(self.kf.x[0])
        direction = -1
      
      # y_pred format is [v/(h)]
      y_pred = model.predict([[x_center, int(area)]])

      # restore the velocity in microsecond, and approximate the speed per frame with the average time per frame
      v = y_pred*h*116*direction #116 is an average time segment per frame calculated by the train dataset

      # change the constant velocity in self.kf.x with the predicted velocity v
      # these print commands are used to compare the origin velocity and the predictions
      #print('constant velocity: ', self.kf.x[4])
      #print('velocity/prediction: ',int(self.kf.x[4])/v)
      self.kf.x[4]= v
      #print('predicted velocity: ', v,'\n')
    if average_speed_ini:
      direction = 1 # 1 means from left to right, while -1 means from right to left
      x_center = int(self.kf.x[0])
      
      #find out the velocity direction 
      if (int(self.kf.x[4])==0 and int(self.kf.x[0])>160) or int(self.kf.x[4])<0:
        x_center = 320-int(self.kf.x[0])
        direction = -1
      v = 0.128*116*direction
      self.kf.x[4]= v
      print('average init')


    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1][0]


  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)[0]

  def get_gt(self):
    return self.ground_truth
  def get_id(self):
    return self.id


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area of the boundingbox
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
