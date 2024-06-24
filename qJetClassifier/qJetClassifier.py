from torch import nn
import torch
import pennylane as qml
from qJetClassifier.QLayer import Qlayer, reuploading_circuit
from torch_geometric.utils import to_networkx

class qJetClassifier(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.hidden_layer_size = model_params['hidden_layer_size']
        self.latent_space_size = model_params['latent_space_size']
        self.nb_reuploading = model_params['nb_reuploading']
        self.nb_ansatz_layers = model_params['nb_ansatz_layers']
        self.qdevice = model_params['qdevice']
        self.diff_method = model_params['diff_method']
        self.nb_node_features = model_params['nb_node_features']

        self.l1 = nn.Linear(self.nb_node_features, self.hidden_layer_size, dtype=torch.float64)
        self.l2 = nn.Linear(self.hidden_layer_size, self.latent_space_size, dtype=torch.float64)
        self.qnode = qml.QNode(self._quantum_classifier, self.qdevice, interface="torch", diff_method=self.diff_method)
        self.qlayers = [QLayer(self.qnode, self.nb_reuploading*(self.nb_ansatz_layers*4+self.nb_node_features+self.latent_space_size)) for i in range(5)]

    def forward(self, x):
        edge_attr = x.edge
        edge_attr = edge_attr[:,0], edge_attr[:,1]
        edge_attr = nn.functional.leaky_relu(0.5*(self.l1(edge_attr[0]))+self.l1(edge_attr[1]))

        edge_attr = nn.functional.softmax(self.l2(edge_attr))

        x.edge = edge_attr

        list_graph = x.to_data_list()
        output = []
        for x in list_graph:   #TODO: Is there a better way to do this?
            probabilities = []

            for i in range(5):
                measurement = self.qlayers[i](x)
                measurement = torch.stack(measurement)
                measurement_normalized = torch.sum(measurement)/measurement.shape[0] + 0.5
                probabilities.append(measurement_normalized)
            output.append(torch.stack(probabilities))
        return torch.stack(output)

    def _quantum_classifier(self, inputs, weights):
        """The quantum classifier

        Args:
            inputs (Networkx.Graph): a fully connected graph representing the jet
            weights (torch.Tensor): the trainable parameters for the classifier
            nb_reuploading (int): the number of re-uploading steps
            nb_ansatz_layers (int): the number of layers of the ansatz
        """
        inputs = to_networkx(inputs, to_undirected=True, edge_attrs=['edge'], node_attrs=['pTEtaPhi'])
        probability = reuploading_circuit(inputs, weights, self.nb_reuploading, self.nb_ansatz_layers)
        
        return probability