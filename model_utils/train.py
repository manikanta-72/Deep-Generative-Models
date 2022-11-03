import torch

class InstanceTrainer():

    def __init__(
            self,
            model,
            loss_func,
            optimizer
            ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
    
    def train_one_epoch(self, train_data):
        # set enable model training mode to compute gradients
        self.model.train()
        
        train_loss = []
        for batch_idx, (x, y) in enumerate(train_data):
            # flatten the input 
            x = x.reshape(-1,1)

            # transfer the inputs to device
            x = x.to(self.device)
            # if y is not None:
            #     y = y.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # take a step in training
            p, x_gen = self.model(x)

            # compute loss 
            loss = self.loss_func(p + 1e-10 , x)

            loss.backward()

            # compute gradient and take a step towards objective
            self.optimizer.step()
           
            print("loss: ", loss.item())
            train_loss.append(loss.item())
        
        return train_loss    
