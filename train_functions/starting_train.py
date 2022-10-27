import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook
from torch.utils.tensorboard import SummaryWriter


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    writer = SummaryWriter()

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    train_losses=[]
    step = 1
    for epoch in range(epochs):
        #print(f"Epoch {epoch + 1} of {epochs}")

        progress_bar = tqdm(train_loader, leave=False)
        losses = []
        total = 0
        for inputs, target in progress_bar:
            model.zero_grad() # reset the gradient values from previous iteration
            #print("hello")
            output = model(inputs) # run forward prop 
            loss = loss_fn(output.squeeze(), target.float()) # calculate cost/loss 
            '''DEBUG
            if(total == 0):
                print(inputs.size(),inputs)
                print(target.size(),target)
                print(output.size(),output)
            '''
            loss.backward() # back propagation: computing gradients
            #print("hello1")
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            #print("hello2")
            optimizer.step() # update weights: using gradients to update weights 
            #print("hello3")
            progress_bar.set_description(f'Loss: {loss.item():.3f}') 
            #print("hello4")
            losses.append(loss.item()) # keep track of losses as we iterate 
            total += 1
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # Compute training loss and accuracy.
                #print("hello5")
                train_loss,train_accuracy = evaluate(train_loader,model,loss_fn)
                #print("hello6")
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                val_loss,val_accuracy = evaluate(val_loader, model, loss_fn)
                #print("hello7")
                writer.add_scalar('Loss/train', train_loss, step//n_eval)
                writer.add_scalar('Loss/validation', val_accuracy, step//n_eval)
                writer.add_scalar('Accuracy/train', train_accuracy, step//n_eval)
                writer.add_scalar('Accuracy/validation', val_accuracy, step//n_eval)
                #print("hello8")
                model.train()

            step += 1

        epoch_loss = sum(losses) / total #calculate an overall loss for an epoch 
        train_losses.append(epoch_loss)

        tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    #print(outputs.size(),labels.size())
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    """
    model.eval()
    with torch.no_grad():
        accuracies = []
        losses = []
        num=0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(), labels.float())
            accuracy = compute_accuracy(outputs.squeeze(),labels.float())
            losses.append(loss)
            accuracies.append(accuracy)
            num+=1
    return (sum(losses)/num),(sum(accuracies)/num)
