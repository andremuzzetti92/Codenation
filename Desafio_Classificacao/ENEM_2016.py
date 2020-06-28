# -*- coding: utf-8 -*-
"""
Created on Mon May 25 04:16:48 2020

@author: André Luís Muzzetti Mateus

Criado com intuito de realizar o desafio de MACHINE LEARNING proposto pela plataforma
CODENATION

Neste código realizei uma simples regressão linear utilizando as colunas que julguei importante


Nota: Linhas com ##--## São Títulos
      Linhas com ### são comentários
      
"""

##--## Importando Biblíotecas necessárias ##--##

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
##--## Dados treino cru ##--##

data = pd.read_csv("train.csv")

##--## Criando Lista das colunas úteis ##--##

colunas = [
    'NU_NOTA_MT',
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO']

##--## Criando DataFrame para treino ##--##

train = pd.DataFrame(data[colunas])
train.dropna(inplace=True)
target=train["NU_NOTA_MT"]                         ### Alvo predicao
train.drop("NU_NOTA_MT",axis=1,inplace = True)    ### Excluindo alvo do dataFrame

##--## Splitando os dados para treino e teste do modelo

X=train
y=target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

##--## Criando e treinando modelo ##--##

lm = LinearRegression()                     ### Instanciando modelo
lm.fit(X_train,y_train)                     ### Treinando modelo
predictions = lm.predict(X_test)            ### Previsão modelo 
print(lm.coef_)                             ### Observando os coeficientes de cada variável

##--## Evaluando o modelo

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

##---------------------------------##

##--## Prevendo a variável alvo para o teste ##--##

##--## Obtendo os dados e selecionando as colunas importantes

dados_teste = pd.read_csv("test.csv")
insc = dados_teste['NU_INSCRICAO']
insc = pd.DataFrame(insc)
colunas_teste = [
    'NU_NOTA_CN',
    'NU_NOTA_CH',
    'NU_NOTA_LC',
    'NU_NOTA_REDACAO',
    ]

teste = pd.DataFrame(dados_teste[colunas_teste])
teste.dropna(inplace=True)

##--## Utilizando o modelo criado para prever ##--##

teste_pred = lm.predict(teste)
nota = pd.DataFrame(teste_pred,columns=['NU_NOTA_MT'],index=teste.index)


