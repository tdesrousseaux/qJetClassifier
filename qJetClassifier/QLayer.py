from typing import Dict
import pennylane as qml
import torch
from torch import nn
import contextlib

def R_X_layer(graph, parameters):
    """Encode the nodes of the graph using R_X gates, multiplied by trainable parameters

    Args:
        graph (Networkx.Graph): a fully connected graph representing the jet
        parameters (torch.Tensor): the trainable parameters for the equivariant layer
    """

    for i in range(graph.num_nodes):
        qml.RX(torch.inner(parameters, graph.pTEtaPhi[i]),
               wires=i)
        
def ZZ_layer(graph, parameters):
    """Encode the edges of the graph using ZZ gates, multiplied by trainable parameters

    Args:
        graph (Networkx.Graph): a fully connected graph representing the jet
        parameters (torch.Tensor): the trainable parameters for the equivariant layer
    """

    for i in range(len(graph.edge)):  #The order doesn't matter, the gates commute between each other
        qml.IsingZZ(torch.inner(parameters, graph.edge[i]),
                    wires=graph.edge_index[:,i].tolist())


def feature_map(graph, feature_parameters):
    """Encode the nodes and edges of the graph using R_X and ZZ gates, multiplied by trainable parameters

    Args:
        graph (torch_geometric.data.Data): a fully connected graph representing the jet
        feature_parameters (torch.Tensor): the trainable parameters for the feature map
    """
    nb_node_features = len(graph.pTEtaPhi[0])
    nb_edge_features = len(graph.edge[0])
    nb_parameters = len(feature_parameters)

    assert nb_parameters == nb_node_features + nb_edge_features, "The number of feature parameters is not correct"


    R_X_layer(graph, feature_parameters[:nb_node_features])
    ZZ_layer(graph, feature_parameters[nb_node_features:])

def equivariant_ansatz(graph, parameters, nb_layers):
    """The trainable equivariant ansatz

    Args:
        graph (torch_geometric.data.Data): a fully connected graph representing the jet
        parameters (torch.Tensor): the trainable parameters for the equivariant layer
        nb_layers (int): the number of layers of the ansatz
    """

    nb_wires = graph.num_nodes
    assert len(parameters) == 4 * nb_layers, "The number of parameters of the ansatz is not correct"

    for layer in range(nb_layers):
        for i in range(nb_wires):
            qml.Rot(*parameters[4*layer:4*layer+3], wires=i)
        for i in range(len(graph.edge)):
            qml.IsingZZ(parameters[4*layer+3], wires=graph.edge_index[:,i].tolist())

def reuploading_circuit(graph, parameters, nb_reuploading, nb_ansatz_layers):
    """A circuit that encodes the graph using the re-uploading strategy

    Args:
        graph (torch_geometric.data.Data): a fully connected graph representing the jet
        parameters (torch.Tensor): the trainable parameters for the equivariant layer
        nb_reuploading (int): the number of re-uploading steps
        nb_ansatz_layers (int): the number of layers of the ansatz
    """

    nb_parameters = len(parameters)
    nb_parameters_per_layer = nb_parameters // nb_reuploading
    assert nb_parameters % nb_reuploading == 0, "The number of parameters is not divisible by the number of re-uploading steps"
    num_nodes = graph.num_nodes

    for i in range(num_nodes):
        qml.Hadamard(wires=i)

    for layer in range(nb_reuploading):
        layer_parameters = parameters[layer*nb_parameters_per_layer:(layer+1)*nb_parameters_per_layer]
        ansatz_parameters = layer_parameters[:4*nb_ansatz_layers]
        feature_parameters = layer_parameters[4*nb_ansatz_layers:]

        feature_map(graph, feature_parameters)
        equivariant_ansatz(graph, ansatz_parameters, nb_ansatz_layers)


    probability = [qml.expval(qml.Z(wire)) for wire in range(num_nodes)]          #TO MODIFY
    return probability

class QLayer(nn.Module):
    def __init__(self, qnode, nb_parameters):   
        super(QLayer, self).__init__()
        self.qnode = qnode
        self.nb_parameters = nb_parameters
        self.qnode_weights: Dict[str, torch.nn.Parameter] = {}
        self._init_weights()

    def __getattr__(self, item):
        """If the qnode is initialized, first check to see if the attribute is on the qnode."""
        if self._initialized:
            with contextlib.suppress(AttributeError):
                return getattr(self.qnode, item)

        return super().__getattr__(item)

    def __setattr__(self, item, val):
        """If the qnode is initialized and item is already a qnode property, update it on the qnode, else
        just update the torch layer itself."""
        if self._initialized and item in self.qnode.__dict__:
            setattr(self.qnode, item, val)
        else:
            super().__setattr__(item, val)

    def _init_weights(self):
        self.qnode_weights["weights"] = nn.Parameter(torch.randn(self.nb_parameters, dtype=torch.float64))
        self.register_parameter("weights", self.qnode_weights["weights"])

    def forward(self, graph):
        kwargs = {
            **{self.input_arg: graph},
            **{arg: weight for arg, weight in self.qnode_weights.items()},
        }
        
        return self.qnode(**kwargs)
    
    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)
    
    __repr__ = __str__

    _input_arg = "inputs"
    _initialized = False


    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg
    
    @staticmethod
    def set_input_argument(input_name: str = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        QLayer._input_arg = input_name

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        x = args[0]
        kwargs = {
            self.input_arg: x,
            **{arg: weight.data for arg, weight in self.qnode_weights.items()},
        }
        self.qnode.construct((), kwargs)