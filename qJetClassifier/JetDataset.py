from torch.utils.data import Dataset
import numpy as np  
from torch_geometric.utils import from_networkx, to_networkx


class JetDataset(Dataset):

    def __init__(self, x_file_path: str, y_file_path: str, n_elements=750):
        """

        Args:
            x_file_path (string): Path to the file with the x data.
            y_file_path (string): Path to the file with the y data.
            n_elements (int): Number of elements to take from the dataset
        """
        self.x_file_path = x_file_path
        self.y_file_path = y_file_path
        self.n_elements = n_elements
        self.data = self.__convert_to_graph(self.__load_data(x_file_path, y_file_path))

    def __len__(self):
        """Returns the length of the dataset
        
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the item at the given index
        
        Args:
            idx (int): Index of the item to return
            
        Returns:
        """
        graph = self.data[idx][0]
        label = self.data[idx][1]
        graph = from_networkx(graph)
        graph.y = label
        return graph
    
    def __load_data(self, x_file_path: str, y_file_path: str):
        x = np.load(x_file_path)
        # y = np.load(y_file_path).astype(np.int64)
        y = np.load(y_file_path)
        indices_list = np.random.choice(x.shape[0], self.n_elements, replace=False)
        return ([[x[i], y[i]] for i in indices_list])


    def __convert_to_graph(self, dataset):
        
        graph_dataset = []
        for data_element in dataset:
            G = nx.DiGraph()
            x = data_element[0]
            # print(type(x[0]))
            y = data_element[1]
            nb_particles = x.shape[0]
            for i in range(nb_particles):
                G.add_node(i, pTEtaPhi=x[i])
                for j in range(i+1, nb_particles):
                    G.add_edge(i, j, edge = [x[i], x[j]])

            graph_dataset.append([G,y])
        
        return graph_dataset