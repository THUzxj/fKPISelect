import os
import argparse

from torch.backends import cudnn
from anomaly_detection.solvers import UsadSolver, AnomalyTransformerSolver

from anomaly_detection.utils import get_default_device


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_on_train', 'cluster'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    parser.add_argument('--output_dir')
    parser.add_argument('--output_file')
    parser.add_argument('--event_output_file')
    parser.add_argument('--model_name')
    parser.add_argument('--model_init_checkpoint')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--model_type', choices=['usad', 'transformer', 'bsad', 'dagmm', 'uniad'], default='usad')
    parser.add_argument('--multiple_anomaly_ratios', action='store_true')
    parser.add_argument('--subset_num', type=int)
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.005,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    
    parser.add_argument('--inspect_scores', action='store_true')
    parser.add_argument('--save_input_output', action='store_true')
    parser.add_argument('--save_output', action='store_true')
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard', 'minmax'])
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--select_file', type=str)

    config = parser.parse_args()
    if config.model_name == None:
        config.model_name = '_'.join([config.model_type, config.dataset])

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    return config


if __name__ == "__main__":
    config = getArgs()
    device = get_default_device()
    print(device)

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.makedirs(config.model_save_path, exist_ok=True)

    solver = None
    if config.model_type == "usad":
        solver = UsadSolver(vars(config))
    elif config.model_type == "transformer":
        solver = AnomalyTransformerSolver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'test_on_train':
        solver.test_on_train_data()

    
    if config.save_input_output:
        if config.mode == 'train':
            solver.save_train_input_output()
        elif config.mode == 'test':
            solver.save_test_input_output()
                