def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--Dueling',type=bool,default=False,help='imporved_hw4_2')
    parser.add_argument('--improved',type=bool,default=False,help='imporved_hw4_1')

    return parser
