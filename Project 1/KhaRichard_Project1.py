#!/usr/bin/env python
# coding: utf-8

# Author: Richard Kha
# 
# Email: kharichard@csu.fullerton.edu
# 
# CPSC 483 - 01
# 
# Project 1 - Tukey's Fences
# 
# In this project, we will be using Tukey's fences to identify outliers in terms of interquartile range in order to find students that should not get an attendance grade. We will be finding the outliers in three different weeks of data from a csv file and identify any students outside of the interquartile range. By identifying the outliers, the students that did not fully attend the zoom lecture will be listed along with their times. This program outputs which week it is listing, the quartiles for that week, Tukey's Range for the week, and the students outside the outlier for the week.

# In[105]:


#Part 1
#Load and examine dataset

import csv

#Opens file and reads it
with open('participants.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
      
    #Appends column data of week times into a list
    weeklist = [] 
    for row in reader:
        weeklist.append({'Name': row['Name'], 'Week 1' : row['Week 1'], 'Week 2' : row['Week 2'], 'Week 3' : row['Week 3']})

#Choose which week to use for output formatting
def define_week(week):
    temp = 'Week '
    temp += week
    return temp


# In[106]:


#Part 2
#Find quartile ranges and quartiles for each week

#Count total number of students
count = 0
for x in range(len(weeklist)):
    count = count + 1  
    
#Determines if even or odd amount of students to find quartile intervals
if (count % 2 == 0):
    q1_pos = count / 4
    q2_pos = count / 2
    q3_pos = (count / 2) + q1_pos
else:
    q1_pos = (count / 4) + 1
    q2_pos = (count / 2) + 1
    q3_pos = (count / 2) + q1_pos + 1 
    
    
#Determines quartiles
def find_quartile(quarter, week_number, sort_week):
    for i in range(int(quarter)):
        compare1 = int(sort_week[i][week_number])
        compare2 = int(sort_week[i + 1][week_number])
    ans = (compare1 + compare2) / 2
    return ans


# In[107]:


#Part 2-5
#Set quartiles, tukey's range, and consolidate to tardy function

def tardy():
    #Loop for the three different weeks
    for i in range(1, 4):
        week_number = define_week(str(i))
    
        #Sort the week times
        sort_week = sorted(weeklist, key = lambda x:x[week_number].zfill(3))     

        #Set quartiles
        q1 = find_quartile(q1_pos, week_number, sort_week)
        q2 = find_quartile(q2_pos, week_number, sort_week)
        q3 = find_quartile(q3_pos, week_number, sort_week)
    
        #Print which week this is
        print ('Data for', week_number)
        print ('---------------')
    
        #Print Quartiles
        print ('Quartile 1:', q1)
        print ('Quartile 2:', q2)
        print ('Quartile 3:', q3)
        print ('-----------------------------')
    
        #Use Tukey's Fences Method to interquartile range
        k = 1.5
        tukey_1 = q1 - k * (q3 - q1)
        tukey_2 = q3 + k * (q3 - q1)
    
        #Print Tukey's Range
        print ("Tukey's Range: [", tukey_1, ',', tukey_2, "]")
        print ('-----------------------------')
    
        #List of outliers
        outliers = []
    
        #Finds outliers and appends them to the outlier list
        for i in range(len(weeklist)):
            if (tukey_1 > float(weeklist[i][week_number])):
                outliers.append(weeklist[i][week_number])
        
        #Get data of the outliers and append full data into a list
        list = []
        for row in weeklist:
            for i in range(len(outliers)):
                if outliers[i] == row[week_number]:
                    list.append({'Name': row['Name'], 'Week 1' : row['Week 1'], 'Week 2' : row['Week 2'], 'Week 3' : row['Week 3']})
            
        #Print the outlier students
        print ('These are the students with questionable attendance times.')
        for i in range(len(list)):
            print (list[i])
        print ('\n')


# In[108]:


#Use tardy function
tardy()


# In[ ]:




