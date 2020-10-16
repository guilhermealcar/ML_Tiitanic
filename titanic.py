import pandas as pd
from sklearn import tree
import statistics
from sklearn import linear_model

#Abre os arquivos de treino e teste
df_treino = pd.read_csv('train.csv')
df_teste = pd.read_csv('test.csv')

#Substitui homens e mulheres por 0 e 1, respectivamente
df_treino['Sex'] = df_treino['Sex'].map({'male': 0, 'female': 1})
df_teste['Sex'] = df_teste['Sex'].map({'male': 0, 'female': 1})

#Converte os valores das portas em que se embarcou
df = [df_treino, df_teste]

for i in df:
    portas = {'S':0, 'C':1, 'Q':2}
    i['Embarked'] = i['Embarked'].map(portas)
    
#Converte os valores desconhecidos das idades pela mediana das idades
df_treino['Age'] = df_treino['Age'].fillna(df_treino['Age'].median())
df_teste['Age'] = df_teste['Age'].fillna(df_teste['Age'].median())

#Converte as portas desconhecidas pela moda das portas
df_treino['Embarked'] = df_treino['Embarked'].fillna(statistics.mode(df_treino['Embarked']))
df_teste['Embarked'] = df_teste['Embarked'].fillna(statistics.mode(df_teste['Embarked']))

#Converte as tarifas desconhecidas para a média das tarifas
df_treino['Fare'] = df_treino['Fare'].fillna(df_treino['Fare'].mean())
df_teste['Fare'] = df_teste['Fare'].fillna(df_teste['Fare'].mean())

#Vetores de input com caracteristicas uteis
X_treino = df_treino.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y_treino = df_treino['Survived'].values
X_teste = df_teste.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#Cria a arvore de decisao
clf_arvore = tree.DecisionTreeClassifier()
clf_arvore = clf_arvore.fit(X_treino, y_treino)

#Plota a árvore de decisão
tree.plot_tree(clf)

#Calcula a acuracia do modelo
acuracia_arvore = clf_arvore.score(X_treino, y_treino)
print('Acuracia da arvore:', round(acuracia_arvore * 100, 2))

#### Perceptron ####
#Cria o perceptron
clf_perceptron = linear_model.Perceptron()
clf_perceptron = clf_perceptron.fit(X_treino, y_treino)

#Calcula a acuracia do modelo
acuracia_perceptron = clf_perceptron.score(X_treino, y_treino)
print('Acuracia do perceptron:', round(acuracia_perceptron * 100, 2))
