### Custom definitions and classes if any ###
import math
import pandas as pd
import numpy as np
import pickle
# C:/Users/chand/Desktop/hackathon2/input_test_data.csv
# input_test_data.csv

def predictRuns(testInput):

  d1={'Eden Gardens': 5081, 'M Chinnaswamy Stadium': 5025, 
    'Wankhede Stadium': 4727, 'Arun Jaitley Stadium': 4599, 
    'MA Chidambaram Stadium': 3805, 'Narendra Modi Stadium': 814}

  d2={'Kolkata Knight Riders': 4078, 'Mumbai Indians': 3868,
    'Royal Challengers Bangalore': 3850, 'Delhi Capitals': 3660,
      'Chennai Super Kings': 3470, 'Punjab Kings': 1953, 
      'Rajasthan Royals': 1878, 'Sunrisers Hyderabad': 1294}

  d3={'Kolkata Knight Riders': 4072, 'Royal Challengers Bangalore': 3860,
    'Mumbai Indians': 3854, 'Delhi Capitals': 3612, 'Chennai Super Kings': 3474,
    'Punjab Kings': 1957, 'Rajasthan Royals': 1922, 'Sunrisers Hyderabad': 1300}
  prediction = 0
  model=pickle.load(open('hackathon.pkl','rb'))
  data=pd.read_csv('useful_data.csv')
  df = pd.read_csv(testInput)
  df=list(df.values)
  # print(df)
  innings=df[0][1]
  venue=d1[df[0][0]]
  batting_team=d2[df[0][2]]
  bowling_team=d3[df[0][3]]

  runs=data[(data['innings']==innings) & (data['ball']<6.1) & (data['batting_team']==batting_team)
  & (data['bowling_team']==bowling_team)]

  matches=math.floor(len(runs['total_runs'])/36)
  score=sum(runs['total_runs'])/matches

  # own prediction
  ball=[0.1,0.2,0.3,0.4,0.5,0.6,1.1,1.2,1.3,1.4,1.5,1.6,2.1,2.2,2.3,
  2.4,2.5,2.6,3.1,3.2,3.3,3.4,3.5,3.6,4.1,4.2,4.3,4.4,4.5,4.6,5.1,5.2,5.3,5.4,5.5,5.6]
  total_runs=0
  for i in ball:
    scores=model.predict([[venue,innings,i,batting_team,bowling_team]])
    total_runs=total_runs+abs(scores)

  return math.floor((0.98*score)+(0.02)*int(total_runs))