import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

alpha = 1

# Reading the data
x_train = pd.read_csv('x_train.csv', sep=' ')
x_val = pd.read_csv('x_test.csv', sep=' ')
y_train = pd.read_csv('y_train.csv', sep='\t', header=None).to_numpy()
y_val = pd.read_csv('y_test.csv', sep='\t',header=None).to_numpy()

# Number of documents classified as Bussiness (0), Entertainment (1), Politics (2), Sport (3) and Tech (4) will be in these arrays
classes_train = np.array([0, 0, 0, 0, 0])
classes_val = np.array([0, 0, 0, 0, 0])

# Counting number of documents for each class
for val in y_train:
    classes_train[val] += 1
    
for val in y_val:
    classes_val[val] += 1

# Pie Charts for Question 3.1 1
#classes = ['Bussiness', 'Entertainment', 'Politics', 'Sport', 'Tech']
#plt.pie(classes_train, labels=classes, autopct='%1.0f%%')
#plt.show()

#classes = ['Bussiness', 'Entertainment', 'Politics', 'Sport', 'Tech']
#plt.pie(classes_val, labels=classes, autopct='%1.0f%%')
#plt.show()

# Calculating log of prior class probabilities
bussiness_prior = np.log(classes_train[0]/np.sum(classes_train))
entertainment_prior = np.log(classes_train[1]/np.sum(classes_train))
politics_prior = np.log(classes_train[2]/np.sum(classes_train))
sport_prior = np.log(classes_train[3]/np.sum(classes_train))
tech_prior = np.log(classes_train[4]/np.sum(classes_train))

print("\nQuestion 3.1 2")
print("P(Y = Bussiness) = " + str(round(classes_train[0]/np.sum(classes_train), 3)))
print("P(Y = Entertainment) = " + str(round(classes_train[1]/np.sum(classes_train), 3)))
print("P(Y = Politics) = " + str(round(classes_train[2]/np.sum(classes_train), 3)))
print("P(Y = Sport) = " + str(round(classes_train[3]/np.sum(classes_train), 3)))
print("P(Y = Tech) = " + str(round(classes_train[4]/np.sum(classes_train), 3)))

# Calcuating word probabilites for each class
bussiness_word_counts = x_train.loc[y_train == 0].select_dtypes(np.number).sum()
entertainment_word_counts = x_train.loc[y_train == 1].select_dtypes(np.number).sum()
politics_word_counts = x_train.loc[y_train == 2].select_dtypes(np.number).sum()
sport_word_counts = x_train.loc[y_train == 3].select_dtypes(np.number).sum()
tech_word_counts = x_train.loc[y_train == 4].select_dtypes(np.number).sum()

bussiness_total_words = x_train.loc[y_train == 0].select_dtypes(np.number).values.sum()
entertainment_total_words = x_train.loc[y_train == 1].select_dtypes(np.number).values.sum()
politics_total_words = x_train.loc[y_train == 2].select_dtypes(np.number).values.sum()
sport_total_words = x_train.loc[y_train == 3].select_dtypes(np.number).values.sum()
tech_total_words = x_train.loc[y_train == 4].select_dtypes(np.number).values.sum()

# Calculations for Question 3.1 4
print("\nQuestion 3.1 4")
alien_word_count = x_train.loc[y_train == 4].select_dtypes(np.number)['alien'].sum()
print("P(alien | Y = Tech) = " + str(alien_word_count/tech_total_words))
print("ln(P(alien | Y = Tech)) = " + str(math.log(alien_word_count/tech_total_words)))
thunder_word_count = x_train.loc[y_train == 4].select_dtypes(np.number)['thunder'].sum()
print("P(thunder | Y = Tech) = " + str(thunder_word_count/tech_total_words))
print("ln(P(alien | Y = Tech)) = -inf")


temp = bussiness_word_counts/bussiness_total_words
temp = np.where((temp)!=0, temp, -math.pow(10,12))
bussiness_word_prior = np.log(temp, out=temp, where=temp>0)
temp = entertainment_word_counts/entertainment_total_words
temp = np.where((temp)!=0, temp, -math.pow(10,12))
entertainment_word_prior = np.log(temp, out=temp, where=temp>0)
temp = politics_word_counts/politics_total_words
temp = np.where((temp)!=0, temp, -math.pow(10,12))
politics_word_prior= np.log(temp, out=temp, where=temp>0)
temp = sport_word_counts/sport_total_words
temp = np.where((temp)!=0, temp, -math.pow(10,12))
sport_word_prior = np.log(temp, out=temp, where=temp>0)
temp = tech_word_counts/tech_total_words
temp = np.where((temp)!=0, temp, -math.pow(10,12))
tech_word_prior = np.log(temp, out=temp, where=temp>0)

# Label Prediction for Question 3.2
mle = np.zeros((y_val.shape[0], 5))
mle_count = 0

for i in range(x_val.shape[0]):
    row = x_val.iloc[i].values
    
    temp_bussiness = bussiness_prior + np.sum(bussiness_word_prior * row)
    temp_entertainment = entertainment_prior + np.sum(entertainment_word_prior * row)
    temp_politics = politics_prior + np.sum(politics_word_prior * row)
    temp_sport = sport_prior + np.sum(sport_word_prior * row)
    temp_tech = tech_prior + np.sum(tech_word_prior * row)
    
    mle[mle_count][0] = temp_bussiness
    mle[mle_count][1] = temp_entertainment
    mle[mle_count][2] = temp_politics
    mle[mle_count][3] = temp_sport
    mle[mle_count][4] = temp_tech
    mle_count += 1      
       
def calculateAccuracy(predictions, true_values):
    confusion_matrix = np.zeros((5,5))
    true_pred = 0
    false_pred = 0
    count = 0
    
    for x in predictions:
        max_index = np.argmax(x)
        
        if max_index == true_values[count]:
            confusion_matrix[max_index][max_index] += 1
            true_pred += 1
        else:
            confusion_matrix[max_index][true_values[count]] += 1
            false_pred += 1
        count += 1
        
    print("Confusion matrix")
    print(confusion_matrix)
    print('Accuracy: ' + str(round(true_pred/(true_pred+false_pred), 3)))

    
# Answer for Question 3.2
print("\nQuestion 3.2")
calculateAccuracy(mle, y_val)

# Calcuating word probabilites for each class
temp = (bussiness_word_counts + 1) / (bussiness_total_words + x_train.shape[1])
temp = np.where((temp)!=0, temp, -math.pow(10,12))
bussiness_word_prior = np.log(temp, out=temp, where=temp>0)
temp = (entertainment_word_counts + 1) / (entertainment_total_words + x_train.shape[1])
temp = np.where((temp)!=0, temp, -math.pow(10,12))
entertainment_word_prior = np.log(temp, out=temp, where=temp>0)
temp = (politics_word_counts + 1) / (politics_total_words + x_train.shape[1])
temp = np.where((temp)!=0, temp, -math.pow(10,12))
politics_word_prior= np.log(temp, out=temp, where=temp>0)
temp = (sport_word_counts + 1)/ (sport_total_words + x_train.shape[1])
temp = np.where((temp)!=0, temp, -math.pow(10,12))
sport_word_prior = np.log(temp, out=temp, where=temp>0)
temp = (tech_word_counts + 1) / (tech_total_words + x_train.shape[1])
temp = np.where((temp)!=0, temp, -math.pow(10,12))
tech_word_prior = np.log(temp, out=temp, where=temp>0)

# Label Prediction for Question 3.3
mle = np.zeros((y_val.shape[0], 5))
mle_count = 0

for i in range(x_val.shape[0]):
    row = x_val.iloc[i].values
    
    temp_bussiness = bussiness_prior + np.sum(bussiness_word_prior * row)
    temp_entertainment = entertainment_prior + np.sum(entertainment_word_prior * row)
    temp_politics = politics_prior + np.sum(politics_word_prior * row)
    temp_sport = sport_prior + np.sum(sport_word_prior * row)
    temp_tech = tech_prior + np.sum(tech_word_prior * row)
    
    mle[mle_count][0] = temp_bussiness
    mle[mle_count][1] = temp_entertainment
    mle[mle_count][2] = temp_politics
    mle[mle_count][3] = temp_sport
    mle[mle_count][4] = temp_tech
    mle_count += 1  
 
# Answer for Question 3.3   
print("\nQuestion 3.3")
calculateAccuracy(mle, y_val)

x_train[x_train > 0] = 1
bussiness_word_counts = x_train.loc[y_train == 0].select_dtypes(np.number).sum()
entertainment_word_counts = x_train.loc[y_train == 1].select_dtypes(np.number).sum()
politics_word_counts = x_train.loc[y_train == 2].select_dtypes(np.number).sum()
sport_word_counts = x_train.loc[y_train == 3].select_dtypes(np.number).sum()
tech_word_counts = x_train.loc[y_train == 4].select_dtypes(np.number).sum()

bussiness_word_prior = (bussiness_word_counts + alpha) / (classes_train[0] + 2 * alpha)
entertainment_word_prior = (entertainment_word_counts + alpha) / (classes_train[1] +  2 * alpha)
politics_word_prior = (politics_word_counts + alpha) / (classes_train[2] + 2 * alpha)
sport_word_prior = (sport_word_counts + alpha)/ (classes_train[3] +  2 * alpha)
tech_word_prior = (tech_word_counts + alpha) / (classes_train[4] +  2 * alpha)

# Label Prediction for Question 3.4
mle = np.zeros((y_val.shape[0], 5))
mle_count = 0

for i in range(x_val.shape[0]):
    row = x_val.iloc[i].values
    row[row > 0] = 1
    temp = (bussiness_word_prior * row) + ((1 - row) * (1 - bussiness_word_prior))
    temp = np.where((temp)!=0, temp, -math.pow(10,12))
    temp_bussiness = bussiness_prior + np.sum(np.log(temp, out=temp, where=temp>0))
    
    temp = (entertainment_word_prior * row) + ((1 - row) * (1 - entertainment_word_prior))
    temp = np.where((temp)!=0, temp, -math.pow(10,12))
    temp_entertainment = entertainment_prior + np.sum(np.log(temp, out=temp, where=temp>0))
    
    temp = (politics_word_prior * row) + ((1 - row) * (1 - politics_word_prior))
    temp = np.where((temp)!=0, temp, -math.pow(10,12))
    temp_politics = politics_prior + np.sum(np.log(temp, out=temp, where=temp>0))
    
    temp = (sport_word_prior * row) + ((1 - row) * (1 - sport_word_prior))
    temp = np.where((temp)!=0, temp, -math.pow(10,12))
    temp_sport = sport_prior + np.sum(np.log(temp, out=temp, where=temp>0)) 
    
    temp = (tech_word_prior * row) + ((1 - row) * (1 - tech_word_prior))
    temp = np.where((temp)!=0, temp, -math.pow(10,12))
    temp_tech = tech_prior + np.sum(np.log(temp, out=temp, where=temp>0))

    mle[mle_count][0] = temp_bussiness
    mle[mle_count][1] = temp_entertainment
    mle[mle_count][2] = temp_politics
    mle[mle_count][3] = temp_sport
    mle[mle_count][4] = temp_tech
    mle_count += 1  
    
# Answer for Question 3.4  
print("\nQuestion 3.4")
calculateAccuracy(mle, y_val)