
class predictor():

    def __init__(model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(self.device)

    def predict(self, x):
        
        with torch.no_grad():
            # enable evaluation mode for model
            self.model.eval()

            # load the inputs to device
            x = x.to(self.device)

            pred = self.model(x)

        return pred
