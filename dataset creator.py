import numpy as np
import pandas as pd
import requests
import datetime as dt

train_set = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\hly532.csv", on_bad_lines='skip') #importing weather data

weather = train_set.iloc[661800:692520,:] #getting the rows with dtaes that line up with the dublin bikes data

print(weather)

#importing dublin bikes data

first = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\dublinbikes_20211001_20220101.csv", on_bad_lines='skip')

print(first['TIME'])


first['TIME'] =  pd.to_datetime(first['TIME'])#, format='%d-%b-%Y %H:%M:%S') #transforming time column to datetime format

first['TIME'] = first['TIME'].dt.floor('Min')    #getting rid of the seconds counter                             
print(first['TIME'])

first1 = first.sort_values('TIME') #sorting by time so that stations bikes available at each time can be added

print(first1)
print(first1.iloc[0,6])
jvec = []
timevec = []
j = first1.iloc[0,6]

for i in range(1,len(first1.index)): #going from top of the sorted df to the bottom if the time on current row is the same as
    print(i) 
    if first1.iloc[i,1] == first1.iloc[i-1,1]: #the previous one then its bikes available are added to the total bikes available
        j = j+first1.iloc[i,6] #at that time
    else: #if the times are not the same, the total bikes availavble at this time is saved to a list, the time is saved to another
        jvec.append(j) #list of equal length and the bike counter is reset 
        timevec.append(first1.iloc[i-1,1])
        j = first1.iloc[i,6]

df = pd.DataFrame(list(zip(jvec, timevec)),columns =['av','TIME']) #create dataframe from the new lists

#df.rename(columns={"0": "av"}, inplace=True)
#df.rename(columns={"1": "TIME"}, inplace=True)

#print(df)

df.to_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\one.csv") #write this df to csv since the code to here takes a while to run

second = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\one.csv", on_bad_lines='skip') #load the dataframe back


#print(second['TIME'])
second['TIME'] =  pd.to_datetime(second['TIME'],format='%Y-%m-%d %H:%M:%S') #convert the time column to the correct format

print(second)

second["bikes"] = int(0) #create a column of zeros for the number of bikes in use

second = second.sort_values('TIME') #sort by time
print(second['TIME'])
k = 1
l = 0

#This loop finds the number of bikes in use ech 5 min interval
for i in range(0,len(second.index)): #for each 5 min interval
    #print(i)
    if second.iloc[i,2].day == k: #if the day is k, same day numbers from different months are not involved here since the df is sorted by date
        if second.iloc[i,1] > l: #if there are more bikes available than currently in  counter l
            l = second.iloc[i,1] #update the counter l
            print(l)  
            
        second.iloc[i,3] = l - second.iloc[i,1] #add the the bikes column the number of bikes available = (max number of bikes previously avaialbe this day - bikes available now)
    else: #if it is not that day
        k = second.iloc[i,2].day #update day counter
        l = 0 #reset l

# Now that we have the number of correct bikes in use for every 5 minute interval after when the maxiumum number of bikes were in use each day
# We need to get the maxiumum number before this time, assuming there were the same number of bikes in service as at the max
second = second.sort_values('TIME', ascending = False) #inverting the order of the df
k = 31 #starting count from the largest possible day for a month
l = 0

#This code is identical to the loop above except for 2 lines
for i in range(0,len(second.index)):
    #print(i)  
    if second.iloc[i,2].day == k:
        if second.iloc[i,1] > l:
            l = second.iloc[i,1]
            print(l)
        if (l - second.iloc[i,1]) > second.iloc[i,3]: #if the new munber of bikes in use is greater than the current entry
            second.iloc[i,3] = l - second.iloc[i,1] #set it to the larger entry
    else:
        k = second.iloc[i,2].day
        l = 0
        
second = second.sort_values('TIME') #sort the df again back to earliest->latest    
        
  
#p =  second.iloc[2,2]-second.iloc[1,2] 
#print(p)    
second.to_csv("C:\\Users\\User\\Documents\\PostGrad\\ML\\group proj\\two.csv") #write to csv since this step takes a while

#second = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\two.csv", on_bad_lines='skip')
train_set = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\hly532.csv", on_bad_lines='skip') #importing weather data
print(train_set['date'])
train_set['date'] =  pd.to_datetime(train_set['date'],format='%d/%m/%Y %H:%M')
weather = train_set.iloc[661800:692520,:] #getting the rows with dtaes that line up with the dublin bikes data     

#print(weather['date'])
  
#print(weather.iloc[2249,0]) 


#print("HERE")



#print(first.iloc[0,1],first.iloc[1937791,1])

#finding the bounds of the dates in the dataset in the weather df
print(weather.iloc[30719,0])   
#EDIT HERE
weather = weather.iloc[28510:30719,:] #Why write efficient code when you can just run it over less of the dataset




 




#weather.to_csv("C:\\Users\\User\\Documents\\PostGrad\\ML\\group proj\\weather.csv")
#weather = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\weather.csv", on_bad_lines='skip')







second["temp"] = int(0) #creating new coulumn for temperature
second["rain"] = int(0)#creating new column for rainfall 
second["wind"] = int(0) #creating new column for windspeed
second["newtime"] = second["TIME"] #creating new column with just the hour for use in upcoming loop, since the weather data is hourly
second["newtime"] = second["newtime"].dt.floor('H') #limiting new coumn to hour as smallest unit

second['newtime'] =  pd.to_datetime(second['newtime'],format='%d/%m/%Y %H:%M') #make sure the new coumn is the correct format
type(second["newtime"]) #checking type of new column
#print(weather['date'])
#print(second['newtime'])
#weather['date'] =  pd.to_datetime(train_set['date'],format='%d/%m/%Y %H:%M')
second["newtime"] = second["newtime"].dt.floor('H') #limiting new coumn to hour as smallest unit
#print(weather.iloc[175,0], second.iloc[1999,7])
#if weather.iloc[175,0].day == second.iloc[1999,7].day and weather.iloc[175,0].month == second.iloc[1999,7].month and weather.iloc[175,0].year == second.iloc[1999,7].year and weather.iloc[175,0].hour == second.iloc[1999,7].hour:
#    print("TRUE")

#This loop matches the weather each hour to the time in df, theres definatley more computationally efficient ways of doing it, but this works
for i in range(0,len(second.index),12): #taking every 12th entry in the df since there are 12 5 min intervals in an hour, makes the loop run much quicker
    #print(i)
    for j in range(0,len(weather.index)): #for each column in the shorter weather
        #print(j)
        if second.iloc[i,7] == weather.iloc[j,0]:#if the time in df is the same as the time in the weather df
            #.year and second.iloc[i,2].month == weather.iloc[i,0].month and second.iloc[i,2].day == weather.iloc[i,0].day and second.iloc[i,2].hour == weather.iloc[i,0].hour:
            print(i)
            second.iloc[i,4] = weather.iloc[j,4] #make the temperature match
            second.iloc[i,5] = weather.iloc[j,2] #make the rainfall match
            second.iloc[i,6] = weather.iloc[j,12] #make the windspeed match


#We now need to expand these weather readings which are in 1 row for each hour to the 12 other rows
for i in range(1,len(second.index)): #for the df
    if second.iloc[i,7] == second.iloc[i-1,7] and second.iloc[i,4] == 0: #if it is the same hour and the row currently doesnt have weather readings
        second.iloc[i,4] = second.iloc[i-1,4] #make the temperature match
        second.iloc[i,5] = second.iloc[i-1,5] #make the rainfall match
        second.iloc[i,6] = second.iloc[i-1,6] #make the windspeed match
    
second = second.sort_values('TIME', ascending = False) #now we have the correct weather in each row each hour below the one initailly filled in, so we inver the df

#and run the same loop again filling in the empty rows
for i in range(1,len(second.index)):
    if second.iloc[i,7] == second.iloc[i-1,7] and second.iloc[i,4] == 0:
        second.iloc[i,4] = second.iloc[i-1,4]
        second.iloc[i,5] = second.iloc[i-1,5]
        second.iloc[i,6] = second.iloc[i-1,6]
        
second = second.sort_values('TIME') #sorting the dataframe from earlier to later again




second.to_csv("C:\\Users\\User\\Documents\\PostGrad\\ML\\group proj\\two.csv") #saving to csv





#first['TIME'].dtypes
#d/m/Y h:m
#match weather to 















