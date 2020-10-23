import pandas as pd
import statistics


def preprocessing(path='', filename='train.csv', rescale=False):
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
    X = df_treino.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df_treino['Survived'].values
    
    return(X, y)