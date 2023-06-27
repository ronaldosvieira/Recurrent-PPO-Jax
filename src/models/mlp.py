import jax.numpy as jnp
from flax import linen as nn


class FeedForward(nn.Module):
    d_model: int
    n_layers: int

    @nn.compact
    def __call__(self, inputs,terminations,last_state):
        for i in range(self.n_layers):
            inputs = nn.Dense(self.d_model)(inputs)
            inputs = nn.relu(inputs)
        return inputs, last_state

    @staticmethod
    def initialize_state(**kwargs):
        #Return a dummy hidden state so that the model can be initialized
        return (jnp.zeros((10,)),)