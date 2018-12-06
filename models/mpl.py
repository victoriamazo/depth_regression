import torch.nn as nn
from collections import OrderedDict


class mpl(nn.Module):
    def __init__(self, FLAGS):
        super(mpl, self).__init__()
        self.input_dims = FLAGS.width * FLAGS.height
        assert isinstance(self.input_dims, int), 'Please provide int for input_dims'
        self.n_hiddens = list(map(int, FLAGS.n_hiddens.split(',')))    #[int(n_hid) for n_hid in FLAGS.n_hiddens.split(' ')]
        self.num_classes = FLAGS.num_classes
        if hasattr(FLAGS, 'keep_prob'):
            self.keep_prob = FLAGS.keep_prob
        else:
            self.keep_prob = 1.0

        current_dims = self.input_dims
        layers = OrderedDict()

        if isinstance(self.n_hiddens, int):
            self.n_hiddens = [self.n_hiddens]
        else:
            self.n_hiddens = list(self.n_hiddens)

        for i, n_hidden in enumerate(self.n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(1-self.keep_prob)
            current_dims = n_hidden

        layers['out'] = nn.Linear(current_dims, self.num_classes)

        self.model = nn.Sequential(layers)
        # print(self.model)


    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims

        return self.model.forward(input)