from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from sklearn.decomposition import PCA
import pandas as pd 


def pre_processed(Path):
    df = pd.read_csv(Path)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    features = PCA(n_components=5).fit_transform(features)   # Reducing the number of features
    return (features, labels)

def predict(Path):
    loaded_vqc = VQC.load('QML_Team_H')
    sampler = Sampler()
    loaded_vqc.neural_network.sampler = sampler

    # Make predictions on the test set with the loaded model
    print("loading...\n")
    features, labels= pre_processed(Path)
    predicted_labels = loaded_vqc.predict(features)
    score = loaded_vqc.score(features, labels)
    
    predicted_labels = map(int, list(predicted_labels))
    print(f"Quantum VQC score: {score:.6f}\n")

    return list(predicted_labels)


