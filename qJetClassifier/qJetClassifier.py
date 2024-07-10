from torch import nn
import torch
import pennylane as qml
from qJetClassifier.QLayer import QLayer, reuploading_circuit
from torch_geometric.utils import to_networkx

class qJetClassifier(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.hidden_layer_size = model_params['hidden_layer_size']
        self.latent_space_size = model_params['latent_space_size']
        self.nb_reuploading = model_params['nb_reuploading']
        self.nb_ansatz_layers = model_params['nb_ansatz_layers']
        self.qdevice = qml.device('lightning.qubit', wires=6)
        self.diff_method = model_params['diff_method']
        self.nb_node_features = model_params['nb_node_features']

        self.sum_observable = model_params['sum_observable'] #Define the kind of observable to use
        if self.sum_observable:
            self.observable = self._sum_measurements
        else:
            self.observable = self._product_measurements

        self.l1 = nn.Linear(self.nb_node_features, self.hidden_layer_size, dtype=torch.float64)
        self.l2 = nn.Linear(self.hidden_layer_size, self.latent_space_size, dtype=torch.float64)
        self.qnode = qml.QNode(self._quantum_classifier, self.qdevice, interface="torch", diff_method=self.diff_method)
        self.qlayers = nn.ModuleList([QLayer(self.qnode, self.nb_reuploading*(self.nb_ansatz_layers*4+self.nb_node_features+self.latent_space_size)) for i in range(5)])

    def forward(self, x):
        edge_attr = x.edge
        edge_attr = edge_attr[:,0], edge_attr[:,1]
        # print(((self.l1(edge_attr[0]))+self.l1(edge_attr[1])).shape)
        edge_attr = nn.functional.leaky_relu(0.5*(self.l1(edge_attr[0])+self.l1(edge_attr[1])))
        # print(edge_attr)
        edge_attr = nn.functional.softmax(self.l2(edge_attr), dim=1)
        # print(edge_attr)
        x.edge = edge_attr
        # print(x.edge)
        list_graph = x.to_data_list()
        output = []
        # print("edge: ", list_graph[0].edge, "node: ", list_graph[0].pTEtaPhi)
        for x in list_graph:   #TODO: Is there a better way to do this?
            probabilities = []
            # print(x.edge)
            for i in range(5):
                measurement = self.qlayers[i](x)
                measurement = torch.stack(measurement)
                measurement_normalized = self.observable(measurement)
                probabilities.append(measurement_normalized)
            output.append(torch.stack(probabilities))
            
        # print(output)
        return nn.functional.softmax(torch.stack(output), dim=1)

    def _quantum_classifier(self, inputs, weights):
        """The quantum classifier

        Args:
            inputs (torch_geometric.data.Data): a fully connected graph representing the jet
            weights (torch.Tensor): the trainable parameters for the classifier
            nb_reuploading (int): the number of re-uploading steps
            nb_ansatz_layers (int): the number of layers of the ansatz
        """
        probability = reuploading_circuit(inputs, weights, self.nb_reuploading, self.nb_ansatz_layers)
        # print(probability)
        return probability
    
    def _product_measurements(self, measurements):
        """Compute the product of the measurements
        
        Args:
            measurements (torch.Tensor): the measurements
            
        Returns:
            product (torch.Tensor): the product of the measurements
        """
        return (torch.prod(measurements) + 1)/2

    def _sum_measurements(self, measurements):
        """Compute the sum of the measurements

        Args:
            measurements (torch.Tensor): the measurements
        
        Returns:
            sum (torch.Tensor): the sum of the measurements
        """
        return (torch.sum(measurements)/measurements.shape[0] + 1)/2 