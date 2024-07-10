import numpy as np

def check_repartition(dataloader):
    """Check the repartition of the classes in the dataset.
    
    Args:
        dataloader (DataLoader): DataLoader object containing the dataset.
        
    Returns:
        repartition (list): List containing the number of samples of each class.
        total (int): Total number of samples in the dataset.
    
    """
    repartition = [0,0,0,0,0]
    for _, batch in enumerate(dataloader):
        for i in range(len(batch.y)):
            repartition += batch.y[i]
    return repartition, np.sum(repartition)

def accuracy(output, target):
    """Compute the accuracy of the model

    Args:
        output (torch.Tensor): the output of the model
        target (torch.Tensor): the target of the model
    """
    return (output.argmax(dim=1) == target.argmax(dim=1)).float().mean()