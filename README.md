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
import joblib
import json
import numpy as np

def init():
    global modelo
    modelo = joblib.load("modelo.pkl")

def run(data):
    try:
        dados = np.array(json.loads(data)["data"])
        resultado = modelo.predict(dados)
        return resultado.tolist()
    except Exception as e:
        return str(e)
        from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment(name="env-previsao")
env.python.conda_dependencies = CondaDependencies.create(pip_packages=["scikit-learn", "numpy", "joblib"])
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

inference_config = InferenceConfig(entry_script="score.py", environment=env)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name="meu-endpoint-previsao",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)
print(f"Endpoint: {service.scoring_uri}")
import requests
import json

dados_teste = json.dumps({"data": [[5.1, 3.5, 1.4, 0.2]]})  # Exemplo de entrada

response = requests.post(url="URL_DO_SEU_ENDPOINT", data=dados_teste, headers={"Content-Type": "application/json"})
print(response.json())


from azureml.core.model import Model

model = Model.register(workspace=ws,
                       model_name="meu-modelo-previsao",
                       model_path="modelo/modelo.pkl")
                       
