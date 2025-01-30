# machinelearning
from azureml.core import Workspace

ws = Workspace.create(name="meu-workspace",
                      subscription_id="ID_DA_SUA_ASSINATURA",
                      resource_group="meu-grupo-de-recursos",
                      create_resource_group=True,
                      location="eastus")
                      from azureml.core import Experiment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

experiment = Experiment(workspace=ws, name="experimento-previsao")

# Carregar os dados
df = pd.read_csv("meus_dados.csv")  # Substitua pelo seu dataset
X = df.drop(columns=["target"])

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
import joblib
import os

os.makedirs("modelo", exist_ok=True)
joblib.dump(modelo, "modelo/modelo.pkl")

from azureml.core.model import Model

model = Model.register(workspace=ws,
                       model_name="meu-modelo-previsao",
                       model_path="modelo/modelo.pkl")
                       
