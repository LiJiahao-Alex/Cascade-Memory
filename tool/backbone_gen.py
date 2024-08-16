from torch import nn

from model import modelUnit


class FLinear:
    def __init__(self, kargs=None):
        if kargs is None:
            layer, acti = None, None
        else:
            layer, acti = kargs
        if layer is None:
            self.layer = [784, 512, 256, 128]
        else:
            self.layer = layer

        if acti is None:
            self.acti = nn.LeakyReLU()
        else:
            self.acti = acti

        encoder_list = []
        decoder_list = []
        for i in range(len(self.layer) - 1):
            encoder_list.append(modelUnit.DenseBlock(self.layer[i], self.layer[i + 1], self.acti))

            if i == len(self.layer) - 2:
                decoder_list.append(
                    modelUnit.DenseBlock(self.layer[len(self.layer) - i - 1],
                                         self.layer[len(self.layer) - i - 2],
                                         acti=None)
                )
            else:
                decoder_list.append(
                    modelUnit.DenseBlock(self.layer[len(self.layer) - i - 1],
                                         self.layer[len(self.layer) - i - 2],
                                         self.acti)
                )
        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(*decoder_list)
        self.name = type(self).__name__
        self.latent_shape = (self.layer[-1],)
