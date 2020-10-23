from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from scipy.stats import wilcoxon

#Importa funcao de outro arquivo
from preprocess import preprocessing

#Separa em dados de treino e dados de teste e cria um y_teste
X, y = preprocessing()
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y)

#### Decision tree ####
#Cria a arvore de decisao
clf_arvore = tree.DecisionTreeClassifier()
clf_arvore = clf_arvore.fit(X_treino, y_treino)


print('----Arvore de decisao----')
#Plot da árvore de decisão
#tree.plot_tree(clf_arvore)

#Calcula a acuracia do modelo
acuracia_arvore_treino = clf_arvore.score(X_treino, y_treino)
print('Acuracia da arvore(treino):', round(acuracia_arvore_treino * 100, 2))

acuracia_arvore = clf_arvore.score(X_teste, y_teste)
print('Acuracia da arvore:', round(acuracia_arvore * 100, 2))

#Cross Validation
crossVal_arvore = cross_val_score(clf_arvore, X, y)
print('Cross Validation arvore:', round(crossVal_arvore.mean() * 100, 2))

#Cross Validation 10 folds
crossVal_arvore_10 = cross_val_score(clf_arvore, X, y, cv=10)
print('Cross Validation arvore com 10 folds:', round(crossVal_arvore_10.mean() * 100, 2))

#Predict
y_pred_arvore = clf_arvore.predict(X_teste)

#Matriz de confusao
cm_arvore = confusion_matrix(y_teste, y_pred_arvore)
#ConfusionMatrixDisplay(cm_arvore).plot()


#### Perceptron ####
print('\n----Perceptron----')

#Cria o perceptron
clf_perceptron = linear_model.Perceptron()
clf_perceptron = clf_perceptron.fit(X_treino, y_treino)

#Calcula a acuracia do modelo
acuracia_perceptron_treino = clf_perceptron.score(X_treino, y_treino)
print('Acuracia do Perceptron(treino):', round(acuracia_perceptron_treino * 100, 2))

acuracia_perceptron = clf_perceptron.score(X_teste, y_teste)
print('Acuracia do Perceptron:', round(acuracia_perceptron * 100, 2))

#Cross Validation
crossVal_perceptron = cross_val_score(clf_perceptron, X, y)
print('Cross Validation Perceptron:', round(crossVal_perceptron.mean() * 100, 2))

#Cross Validation 10 folds
crossVal_perceptron_10 = cross_val_score(clf_perceptron, X, y, cv=10)
print('Cross Validation Perceptron com 10 folds:', round(crossVal_perceptron_10.mean() * 100, 2))

#Predict
y_pred_perceptron = clf_perceptron.predict(X_teste)

#Matriz de confusao
cm_perceptron = confusion_matrix(y_teste, y_pred_perceptron)
#ConfusionMatrixDisplay(cm_perceptron).plot()

#### K Nearest Neighbours ####
print('\n----K Nearest Neighbours----')

clf_knn = KNeighborsClassifier()
clf_knn = clf_knn.fit(X_treino, y_treino)

#Calcula a acuracia do modelo
acuracia_knn_treino = clf_knn.score(X_treino, y_treino)
print('Acuracia da KNN(treino):', round(acuracia_knn_treino * 100, 2))

acuracia_knn = clf_knn.score(X_teste, y_teste)
print('Acuracia da KNN:', round(acuracia_knn * 100, 2))

#Cross Validation
crossVal_knn = cross_val_score(clf_knn, X, y)
print('Cross Validation KNN:', round(crossVal_knn.mean() * 100, 2))

#Cross Validation 10 folds
crossVal_knn_10 = cross_val_score(clf_knn, X, y, cv=10)
print('Cross Validation KNN com 10 folds:', round(crossVal_knn_10.mean() * 100, 2))

#Predict
y_pred_knn = clf_knn.predict(X_teste)

#Matriz de confusao
cm_knn = confusion_matrix(y_teste, y_pred_knn)
#ConfusionMatrixDisplay(cm_knn).plot()

print('\nWilcoxon arvore/perceptron:',wilcoxon(y_pred_arvore, y_pred_perceptron),
      '\n\nWilcoxon arvore/knn:',wilcoxon(y_pred_arvore, y_pred_knn),
      '\n\nWilcoxon perceptron/knn:',wilcoxon(y_pred_perceptron, y_pred_knn))


#Graficos Matplotlib
import matplotlib.pyplot as plt

caracteristicas = ['Acuracia(treino)', 'Acuracia',
                   'CV', 'CV com 10 folds']

fig, ax = plt.subplots()
ax.plot(caracteristicas,
        [round(acuracia_arvore_treino * 100, 2),
         round(acuracia_arvore * 100, 2),
         round(crossVal_arvore.mean() * 100, 2),
         round(crossVal_arvore_10.mean() * 100, 2)],
        label = 'Arvore')

ax.plot(caracteristicas,
        [round(acuracia_perceptron_treino * 100, 2),
         round(acuracia_perceptron * 100, 2),
         round(crossVal_perceptron.mean() * 100, 2),
         round(crossVal_perceptron_10.mean() * 100, 2)],
        label = 'Perceptron')

ax.plot(caracteristicas,
        [round(acuracia_knn_treino * 100, 2),
         round(acuracia_knn * 100, 2),
         round(crossVal_knn.mean() * 100, 2),
         round(crossVal_knn_10.mean() * 100, 2)],
        label = 'KNN')

ax.legend()
fig.suptitle('Relação dos modelos')
plt.show()

