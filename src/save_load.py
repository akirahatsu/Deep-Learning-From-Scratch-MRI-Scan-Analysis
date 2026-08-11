#I BECOME EXTREAMLY LASY FOR THIS PART SO I USE CHAT GPT FOR THIS CODE 


class Model:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def save(self, path="model"):
        
        for i, layer in enumerate(self.layers):

            if hasattr(layer, "weight"):
                np.save(f"{path}_weight_{i}.npy", layer.weight)

            if hasattr(layer, "bias"):
                np.save(f"{path}_bias_{i}.npy", layer.bias)

    def load(self, path="model"):

        for i, layer in enumerate(self.layers):

            if hasattr(layer, "weight"):
                layer.weight = np.load(
                    f"{path}_weight_{i}.npy"
                )

            if hasattr(layer, "bias"):
                layer.bias = np.load(
                    f"{path}_bias_{i}.npy"
                )
