# -*- coding:utf-8 -*-
from multiprocessing import Pool
import os
import time
import argparse



def run_cmd(cmd_str, debug):
    print('Running "{}"\n'.format(cmd_str))
    if debug == 0:
        os.system(cmd_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int,
                        required=True, help='the communication port')
    parser.add_argument('--debug', type=int,  default=0,
                        help='whether use debug mode')
    # Procedure related arguments
    parser.add_argument('--procedure', type=str,
                        required=True, help='the training procedure')
    parser.add_argument('--exp', type=str,
                        required=True, help='experiment_name')
    parser.add_argument('--gpus', nargs='+', type=int,
                        required=True, help='gpu indexes')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--img_class', type=str,
                        default='faz', help='the img class(odoc or faz)')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--iters', type=int,
                        default=10, help='Number of iters (default: 20)')
    parser.add_argument('--eval_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--tsne_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--rep_iters', type=int,
                        default=12, help='Number of iters (default: 12)')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for FedProx')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN')
    parser.add_argument('--sort_lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--sort_beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN/FedUniV2')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The hyper parameter for FedLC/FedALALC/FedAPLC/FedUni/FedUniV2')
    parser.add_argument('--init_iters', type=int,
                        default=100, help='Number of iters (default: 100)')
    parser.add_argument('--common_iters', type=int,
                        default=400, help='Number of iters (default: 400)')
    parser.add_argument('--pretrain_iters', type=int,
                        default=150, help='Number of iters (default: 150)')
    parser.add_argument('--model_momentum', type=float, default=0.5,
                        help='The hyper parameter for FedAP/FedAPLC')
    parser.add_argument('--prompt', type=str, default='universal',
                        help='Prompt type for FedUniV2')
    parser.add_argument('--attention', type=str, default='dual',
                        help='Attention type for FedUniV2')
    parser.add_argument('--dual_init', type=str, default='random',
                        help='Dual branch initiallization for FedUniV2')
    parser.add_argument('--label_prompt', type=int, default=0,
                        help='Whether use label prompt for FedUniV2')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--tree_loss_weight', type=float,  default=0.6, help='treeloss_weight')
    parser.add_argument('--strategy', type=str,
                        default='FedAvg', help='Federated learning algorithm (default: FedAvg)')
    parser.add_argument('--img_size', type=int,  default=256,
                    help='h*w')
    parser.add_argument('--amp', type=int,  default=0,
                        help='whether use amp training')
    args = parser.parse_args()

    assert args.img_class in ['odoc', 'faz', 'polyp']
    assert args.procedure in [
        'flower_pCE_2D', 'flower_pCE_MScaleTreeEnergyLoss_ADD', 'flower_pCE_MScaleTreeEnergyLoss_Recurve', 'flower_pCE_TreeEnergyLoss_2D','flower_pCE_2D_GateCRFTreeEnergyLoss_Ours','flower_pCE_2D_GatedCRFLoss_v1'
    ]
    assert len(args.gpus) == 6

    if args.img_class == 'faz':
        root_path = '../data/FAZ_h5'
        num_classes = 2
        in_chns = 1
        mask_dict = {
            'client1': 'scribble_noisy',
            'client2': 'keypoint',
            'client3': 'block',
            'client4': 'box',
            'client5': 'scribble'
        }
    if args.img_class =='odoc':
        root_path = '../data/ODOC_h5'
        num_classes = 3
        in_chns = 3
        mask_dict = {
            'client1': 'scribble',
            'client2': 'scribble_noisy',
            'client3': 'scribble_noisy',
            'client4': 'keypoint',
            'client5': 'block'
        }
    if args.img_class =='polyp':
        root_path = '../data/Polypdata_h5'
        num_classes = 2
        in_chns = 3
        mask_dict = {
            'client1': 'keypoint',
            'client2': 'scribble',
            'client3': 'box',
            'client4': 'block'
        }

    common_cmd = 'python {}.py --root_path {} --num_classes {} --in_chns {} --img_class {} --exp {} --model {} --max_iterations {} --iters {}'.format(
                    args.procedure, root_path, num_classes, in_chns, args.img_class, args.exp, args.model, args.max_iterations, args.iters) \
                + ' --eval_iters {} --tsne_iters {} --batch_size {} --base_lr {} --amp {} --server_address [::]:{} --strategy {} --min_num_clients {}'.format(
                    args.eval_iters, args.tsne_iters, args.batch_size, args.base_lr, args.amp, args.port, args.strategy, len(mask_dict))
    client_args = ['--role server --client client_all --sup_type mask --gpu {}'.format(args.gpus[0])] \
                + ['--role client --cid {} --client {} --sup_type {} --gpu {}'.format(i, client, sup_type, args.gpus[i + 1])
                    for i, (client, sup_type) in enumerate(mask_dict.items())]

    if args.procedure in ['flower_pCE_MScaleTreeEnergyLoss_ADD', 'flower_pCE_MScaleTreeEnergyLoss_Recurve', 'flower_pCE_TreeEnergyLoss_2D','flower_pCE_2D_GateCRFTreeEnergyLoss_Ours','flower_pCE_2D_GateCRFMsacleTreeEnergyLoss_Ours']:
        common_cmd += ' --tree_loss_weight {}'.format(args.tree_loss_weight)
    if args.procedure in ['flower_pCE_2D_GatedCRFLoss_v1']:
        common_cmd += ' --img_size {}'.format(args.img_size)

    if args.strategy == 'FedProx':
        common_cmd += ' --mu {}'.format(args.mu)
    elif args.strategy == 'FedRep':
        common_cmd += ' --rep_iters {}'.format(args.rep_iters)
    elif args.strategy == 'MetaFed':
        common_cmd += ' --lam {} --beta {} --sort_lam {} --sort_beta {} --init_iters {} --common_iters {}'.format(args.lam, args.beta, args.sort_lam, args.sort_beta, args.init_iters, args.common_iters)
    elif args.strategy == 'FedAP':
        sup_type_list_str = ''
        for sup_type in mask_dict.values():
            sup_type_list_str += '{} '.format(sup_type)
        common_cmd += ' --pretrain_iters {} --model_momentum {} --sup_type_list {}'.format(args.pretrain_iters, args.model_momentum, sup_type_list_str)
    elif args.strategy == 'FedAN':
        common_cmd += ' --lam {} --beta {}'.format(args.lam, args.beta)
    elif args.strategy == 'FedLC':
        common_cmd += ' --alpha {} --rep_iters {}'.format(args.alpha, args.rep_iters)
    elif args.strategy == 'FedALALC':
        common_cmd += ' --alpha {} --rep_iters {}'.format(args.alpha, args.rep_iters)
    elif args.strategy == 'FedAPLC':
        sup_type_list_str = ''
        for sup_type in mask_dict.values():
            sup_type_list_str += '{} '.format(sup_type)
        common_cmd += ' --pretrain_iters {} --model_momentum {} --sup_type_list {}'.format(args.pretrain_iters, args.model_momentum, sup_type_list_str)
        common_cmd += ' --alpha {} --rep_iters {}'.format(args.alpha, args.rep_iters)
    elif args.strategy == 'FedUni':
        common_cmd += ' --alpha {} --rep_iters {}'.format(args.alpha, args.rep_iters)
    elif args.strategy == 'FedUniV2' or args.strategy == 'FedUniV2.1':
        common_cmd += ' --alpha {} --beta {} --prompt {} --attention {} --dual_init {} --label_prompt {}'.format(
                    args.alpha, args.beta, args.prompt, args.attention, args.dual_init, args.label_prompt)

    pool = Pool(len(client_args))
    for i in range(len(client_args)):
        pool.apply_async(run_cmd, ['{} {}'.format(common_cmd, client_args[i]), args.debug])
        if args.debug == 0:
            if i == 0:
                time.sleep(15)
            else:
                time.sleep(5)
        else:
            time.sleep(7)

    pool.close()
    pool.join()