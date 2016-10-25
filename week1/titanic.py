import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv', index_col='PassengerId')
data['Sex'].value_counts()['male']

# how many males and females
with open('q1.txt', 'w') as f:
    write_data = str(data['Sex'].value_counts()['male']) + ' ' + str(data['Sex'].value_counts()['female'])
    f.write(write_data)
f.close()

# how many survived
with open('q2.txt', 'w') as f:
    write_data = float(np.sum(data['Survived'])) / float(np.prod(data.shape[0])) * 100
    f.write(str(round(write_data, 2)))
f.close()

# how many passengers were in 1st class
with open('q3.txt', 'w') as f:
    write_data = float(data['Pclass'].value_counts()[1]) / float(np.prod(data.shape[0])) * 100
    f.write(str(round(write_data, 2)))
f.close()

# how old passengers were
with open('q4.txt', 'w') as f:
    write_data = str(data['Age'].mean()) + ' ' + str(data['Age'].median())
    f.write(write_data)
f.close()

# correlation between brothers/sisters and parents/children on a board
corr = np.corrcoef(data['SibSp'], data['Parch'])[0, 1]
cor = pearsonr(data['SibSp'], data['Parch'])
with open('q5.txt', 'w') as f:
    f.write(str(corr))
f.close()

# most popular female name
female = []
for i in xrange(len(data)):
    if 'Miss.' in data['Name'][i+1]:
        if '(' in data['Name'][i+1]:
            female.append(data['Name'][i+1].split('(')[1].split(' ')[0])
        else:
            female.append(data['Name'][i+1].split(' ')[2])
    if 'Mrs.' in data['Name'][i + 1]:
        if '(' in data['Name'][i + 1]:
            female.append(data['Name'][i + 1].split('(')[1].split(' ')[0])
        else:
            female.append(data['Name'][i + 1].split(' ')[2])
word_counter = {}
for word in female:
    if word in word_counter:
        word_counter[word] += 1
    else:
        word_counter[word] = 1
popular_words = sorted(word_counter, key=word_counter.get, reverse=True)
with open('q6.txt', 'w') as f:
    f.write(str(popular_words[0]))
f.close()

# the reason why sbd survive
X = data[['Pclass', 'Fare', 'Age', 'Sex']]
y = data[['Survived']]
Xy = pd.concat([X, y], axis=1)
Xy = Xy.dropna(axis=0)
X = Xy[['Pclass', 'Fare', 'Age', 'Sex']]
X.Sex = X.Sex.replace(to_replace=['male', 'female'], value=[1, 0])
y = Xy[['Survived']]
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
important = clf.feature_importances_
print important
with open('q7.txt', 'w') as f:
    f.write('Fare' + ' Sex')
f.close()
