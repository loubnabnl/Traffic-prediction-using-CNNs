import pandas as pd

"""The idea behind the following transformations is to have a Time Series 
of Volume as a function of the date for each couple (location_name,direction), the date unit will be 1 hour
"""

def preprocess_data(data):
    """data preprocessing (the data has no missing values or outliers)
    we drop the columns we will not need in our model
     -we drop time bin, this information is already in Hour and Minute
     -we will base our model only on location name
     -Our time series will vary by date: hour-day/month/year, we drop day of week column
    we change date format:
     -grouping year month day and hour into one datatime column "Date"
    each Date value is the hour of a given day (day/month/year)
    we sort sum the traffic volume for each date over minutes
     for each location and direction we will have a time series
    where the volume is a function of "Date-Hour"
    """
    data.drop(['Time Bin', 'location_latitude','location_longitude','Day of Week'], inplace=True, axis=1)
    
    date = data[['Year','Month','Day','Hour']]
    data.drop('Month', inplace=True, axis=1)
    data.drop('Day', inplace=True, axis=1)
    data.drop('Hour', inplace=True, axis=1)
    data[['Year']]=pd.to_datetime(date,unit='D')
    data = data.rename(columns={"Year": "Date"})
    
    data.sort_values(['location_name','Date','Minute'],inplace=True)
    
    group = data.groupby(by = ['location_name','Date','Direction'], as_index=False)['Volume'].sum()
    
    #rearrange data
    d = {'location_name': group['location_name'], 'Direction': group['Direction'],'Date': group['Date'],'Volume':group['Volume']}
    new_data = pd.DataFrame(data=d)
    date = new_data.groupby(['location_name','Direction']).size()
    count_u = new_data.groupby(['location_name','Direction']).size().reset_index().rename(columns={0:'count'})
    #40 rows = 40 couples (location, direction)
    #we delete couples who have less than 100 counts
    new_data = count_u[count_u['count'] > 4000] #7 couples were deleted, 34 are left
    return new_data


def Get_Time_Series(data, location, direction, dates=False):
    #location and direction are strings
    extract = data.loc[data.location_name == location][data.Direction == direction]
    volume = extract['Volume']
    if not dates:
        return volume.to_numpy()
    else:
        #we will need this part later for the plots
        dates = extract['Date']
        frame = {'date': dates,'volume': volume} 
        dataf = pd.DataFrame(frame)
        return dataf

def create_data_dict(data):
    """ Create dictionnary of the locations and directions as key (tuple) 
    and the Traffic Volume as value
    """ 
    prep_data = preprocess_data(data)
    data_dict = {}
    couples = prep_data[['location_name','Direction']]
    couples = [tuple(couples.iloc[i]) for i in range(couples.shape[0])]
    volume = []
    for couple in couples:
        location, direction = couple
        volume.append(Get_Time_Series(data, location, direction))
    for i in range(len(volume)):
        key = couples[i] #(location,direction)
        data_dict[key] = volume[i]  
    return data_dict
