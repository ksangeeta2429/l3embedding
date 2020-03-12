import os
import h5py
import pytz
import numpy as np
import datetime
import pandas as pd
from scipy.stats import rankdata

def get_time_values(timestamp, timezone='America/New_York', 
                    map_2_hrs={str(2*i+k): value for k in range(2) for (i, value) in enumerate(range(12))}):
    
    dt = datetime.datetime.utcfromtimestamp(timestamp)
    dt = pytz.UTC.localize(dt)
    dt = dt.astimezone(pytz.timezone(timezone))
    hour_of_the_day = dt.hour
    day_of_the_week = dt.weekday()
    week_of_the_year = dt.isocalendar()[1]
    
    # Get the 2 hour group id
    hr_id = map_2_hrs[str(hour_of_the_day)]
    
    # Get combination of 2 hour id, day of the week and week of the year to enable groupby later
    # This acts as a unique key for a 2-hr window
    day_id = str(hr_id)+'-'+str(day_of_the_week)+'-'+str(week_of_the_year)
    
    return dict(hour_of_the_day=hour_of_the_day,
                day_of_the_week=day_of_the_week,
                week_of_the_year=week_of_the_year,
                day_id=day_id,
                hr_id=hr_id)
                
def get_spl_frame_vector(spl_vector, spl_iterable=[4*k for k in range(20)]):  
    spl_frames = [0.25*sum([spl_vector[i+k] for k in range(4)]) for i in spl_iterable]
    return dict({'spl_frames': spl_frames})
  
def get_2_hr_relevance_dict(row):
    d = {}
    spl_frames_2_hr = np.array(row['spl_frames'])
    unq, unq_indices = np.unique(spl_frames_2_hr, return_index=True)
    total_frames = len(spl_frames_2_hr)
    ranked_spl = rankdata(spl_frames_2_hr, method='min')
    d = {spl_frames_2_hr[i]: ranked_spl[i]/total_frames for i in unq_indices}
    return d

def get_frame_relevance(row):
    relevance_dict = row['relevance_dict_2_hr']
    lst = [relevance_dict[i] for i in row['spl_frames_emb']]
    return lst

def get_relevance_scores(ts, spl_vecs):

    # Get the spl avg value of 4 consecutive values from spl_vector
    spl_arr = np.apply_along_axis(get_spl_frame_vector, 1, spl_vecs)

    # Apply get_time_values() to each element of the timestamp array 
    dt_vectorize = np.vectorize(get_time_values)
    t_arr = dt_vectorize(ts)

    # Convert the dicts obtained above into dataframe and combine them to make aggregation easier
    t_df = pd.DataFrame(list(t_arr))
    spl_df = pd.DataFrame(list(spl_arr)) 
    df = pd.concat([t_df, spl_df], axis=1)
    
    # Round off the SPL values to 2 decimal places
    df['spl_frames'] = df['spl_frames'].apply(lambda x: list(np.around(np.array(x), decimals=2)))
    
    # Group by 2-hour window
    res = df.groupby(['day_id'], as_index = False).agg({'spl_frames': 'sum'}).reset_index() 
    
    # Form a dictionary with spl values mapped to its relevance score in 2-hour window
    res['relevance_dict_2_hr'] = res.apply(get_2_hr_relevance_dict, axis = 1)
    
    final = pd.merge(df, res, on='day_id', how='outer', suffixes=('_emb', '_2_hr'))
    
    # From the 2-hr dictionary, get the relevance score of the 20 embedding frames corresponding to one entry in blob
    final['relevance_spl_frames'] = final.apply(get_frame_relevance, axis = 1)
     
    return final['relevance_spl_frames']