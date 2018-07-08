#Thomson Sampling

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N = 10000
d = 10
ads_selected = []
number_of_ones = [0] * d
number_of_zeros = [0] * d
total_reward = 0
for n in range(0, N):
   ad = 0
   max_random = 0
   for i in range(0, d):
      random_beta = random.betavariate(number_of_ones[i] + 1, number_of_zeros[i] +1)
      if max_random < random_beta :
         max_random = random_beta
         ad = i
   ads_selected.append(ad)
   reward = dataset.values[n, ad]
   if(reward == 1):
      number_of_ones[ad] = number_of_ones[ad] + 1
   else:
      number_of_zeros[ad] = number_of_zeros[ad] + 1
   total_reward = total_reward + reward
   
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()