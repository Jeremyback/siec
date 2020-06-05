# biblioteki ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# TWORZENIE OBIEKTU DATAFRAME NA PODSTAWIE DANYCH Z WYKŁADU -------------------
# tworzenie list, które będą stanowić kolumny docelowej tabeli
leg_lenght = '1.46 1.32 0.9 1.08 0.53 1.39 0.69 0.59 0.71 2.27 1.61 1.7 1.54'.split()
body_height = '0.86 1.47 1.27 0.95 0.65 2.25 2.18 1.89 1.72 1.77 1.61 2.08 2.03'.split()
animal = 'zebra zebra zebra zebra zebra koń koń koń koń żyrafa żyrafa żyrafa żyrafa'.split()

df = pd.DataFrame()
df['Długość nogi'] = leg_lenght
df['Wysokość ciała'] = body_height
df['Zwierzę'] = animal
print(df)
print()


# TRENOWANIE SIECI ------------------------------------------------------------
# podział zbioru na pozycje treningowe i testowe
X = df[['Długość nogi','Wysokość ciała']]
y = df['Zwierzę']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# pętla, która zbada dla jakiego k, czyli jak dużego sąsiedztwa wyniki są
# najlepiej przewidywane
error_rate = []
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# wykres przedstawiający wynik działania powyższej pętli
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
# wskaźnik błędu jest najmniejszy dla k=1 lub k=4. Przyjmijmy 4.

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# WYNIKI ----------------------------------------------------------------------
print(classification_report(y_test,pred))
print()
print(confusion_matrix(y_test,pred))
