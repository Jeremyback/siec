#biblioteki -------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# TWORZENIE OBIEKTU DATAFRAME W OPARCIU O DANE Z TABELI Z WYKŁADU -------------
#tworzenie list, które będą stanowić kolumny
outlook = 'rainy rainy overcast sunny sunny sunny overcast rainy rainy sunny rainy overcast overcast sunny'.split()
temp = 'hot hot hot mild cool cool cool mild cool mild mild mild hot mild'.split()
humidity = 'high high high high normal normal normal high normal normal normal high normal high'.split()
windy = 'False True False False False True True False False False True True False True'.split()
play_golf = 'No No Yes Yes Yes No Yes No Yes Yes Yes Yes Yes No'.split()

#wkładanie powyższych list do obiektu DataFrame
df = pd.DataFrame()
df['Outlook'] = outlook
df['Temperature'] = temp
df['Humidity'] = humidity
df['Windy'] = windy
df['Play golf'] = play_golf

print(df) #zobaczmy jak wygląda nasza tabela
print()


# EKSPLORACJA DANYCH ----------------------------------------------------------
#ten wykres ukazuje kluczowy wpływ 'outlook' na dezycję grać czy nie grać
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Outlook', hue='Play golf')
plt.show()


# ZMIENNE KATEGORYCZNE --------------------------------------------------------
# należy zwrócić uwagę, że nasze zmienne zawierają dane kategoryczne, natomiast
# metody, których tu się używa, wymagają danych liczbowych. Rozłożymy zatem
# poszczegolne cechy na odrębne kolumny i zapiszemy czy występują czy nie
# w postaci zera lub jedynki
outlook_v2 = ['Outlook']
temp_v2 = ['Temperature']
humidity_v2 = ['Humidity']
windy_v2 = ['Windy']
play_golf_v2 = ['Play golf']

final_df = pd.get_dummies(df,columns=outlook_v2,drop_first=True)
final_df = pd.get_dummies(final_df,columns=temp_v2,drop_first=True)
final_df = pd.get_dummies(final_df,columns=humidity_v2,drop_first=True)
final_df = pd.get_dummies(final_df,columns=windy_v2,drop_first=True)
final_df = pd.get_dummies(final_df,columns=play_golf_v2,drop_first=True)

#tak wygląda efek
print(final_df)
print()


# TRENOWANIE SIECI ------------------------------------------------------------
# dzielenie zbioru danych na treningowy i testowy
X = final_df.drop('Play golf_Yes', axis=1)
y = final_df['Play golf_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=51)


# trenowanie sieci w oparciu o model Decision Tree Classifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)

# wyniki
print(classification_report(y_test,pred))
print()
print(confusion_matrix(y_test,pred))
print()


# trenowanie sieci w oparciu o model Random Fores Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

pred = rfc.predict(X_test)

# wyniki
print(classification_report(y_test,pred))
print()
print(confusion_matrix(y_test,pred))
print()

print('Sieć przewiduje słabo, bo zbiór treningowy jest zdecydowanie zbyt mało',
      'liczny.')
