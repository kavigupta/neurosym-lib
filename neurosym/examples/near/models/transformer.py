import torch.nn as nn


class NearTransformer(nn.Module):
    def __init__(
        self, typ, hidden_size, num_head, num_encoder_layers, num_decoder_layers
    ):
        super().__init__()
        self.typ = typ
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(self, x, *, environment):

        print(environment)
        1 / 0
        # x is a tuple of (source, target)
        return self.transformer(source, target)


def transformer_factory(
    hidden_size: int,
    num_head: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
):
    """
    Allows instantiating a RNN module for sequence-to-class tasks, with a given hidden size.

    :param hidden_size: Size of the hidden layer in the RNN.
    """

    def construct_model(typ):
        return NearTransformer(
            hidden_size=hidden_size,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            typ=typ,
        )

    return construct_model
