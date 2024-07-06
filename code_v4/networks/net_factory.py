from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_320, UNet_DS, UNet_CCT, UNet_CCT_3H, UNet_Head, UNet_MultiHead, \
                            UNet_LC, UNet_LC_MultiHead,UNet_LC_MultiHead_Two, UNet_Uni, UNet_UniV2, UNet_UniV3, UNet_UniV4, UNet_UniV5, UNet_Univ5_Ablation, UNet_UniV5_WO_Uni_Prompt, UNet_UniV5_AttentionConcat, UNet_LC_Auxi

from utils.TreeEnergyLoss.lib.models.nets.fcnet import FcnNet
from utils.TreeEnergyLoss.lib.models.nets.treefcn import TreeFCN
from utils.TreeEnergyLoss.lib.models.nets.deeplabv3plus import DeepLabV3Plus
from utils.TreeEnergyLoss.lib.utils.tools.configer import Configer


def net_factory(args, net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_320":
        net = UNet_320(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct_3h":
        net = UNet_CCT_3H(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "unet_head":
        net = UNet_Head(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_multihead":
        net = UNet_MultiHead(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "fcnet":
        config_dict = {
            "data": {
                "num_classes": class_num
            },
            "network": {
                "backbone": "deepbase_resnet101_dilated8",
                "pretrained": None,
                "bn_type": "torchbn",
                "in_chns": in_chns
            }
        }
        net = FcnNet(Configer(config_dict=config_dict)).cuda()
    elif net_type == "treefcn":
        config_dict = {
            "data": {
                "num_classes": class_num
            },
            "network": {
                "backbone": "deepbase_resnet101_dilated8",
                "pretrained": None,
                "stride": 8,
                "bn_type": "torchbn",
                "in_chns": in_chns,
                "business_channel_num": 512,
                "embed_channel_num": 256,
                "block_channel_nums": [256, 512, 1024, 2048],
                "tree_filter_group_num": 16,
            },
            "tree_loss": {
                "params": {
                    "enable_high_level": True
                }
            }
        }
        net = TreeFCN(Configer(config_dict=config_dict)).cuda()
    elif net_type == "deeplabv3plus":
        config_dict = {
            "data": {
                "num_classes": class_num
            },
            "network": {
                "backbone": "deepbase_resnet101_dilated8",
                "multi_grid": [1, 1, 1],
                "pretrained": None,
                "stride": 8,
                "bn_type": "torchbn",
                "in_chns": in_chns
            },
            "tree_loss": {
                "params": {
                    "enable_high_level": True
                }
            }
        }
        net = DeepLabV3Plus(Configer(config_dict=config_dict)).cuda()
    elif net_type == "unet_lc":
        net = UNet_LC(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_lc_auxi":
        net = UNet_LC_Auxi(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_lc_multihead":
        net = UNet_LC_MultiHead(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_lc_multihead_two":
        net = UNet_LC_MultiHead_Two(in_chns=in_chns, class_num=class_num, pcs_num=1, emb_num=args.min_num_clients,
                    client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_uni":
        net = UNet_Uni(in_chns=in_chns, class_num=class_num, client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_univ2":
        net = UNet_UniV2(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid).cuda()
    elif net_type == "unet_univ3":
        net = UNet_UniV3(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size).cuda()
    elif net_type == "unet_univ4":
        net = UNet_UniV4(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size).cuda()
    
    elif net_type == "unet_univ5":
        net = UNet_UniV5(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size).cuda()
    elif net_type == "unet_univ5_ablation":
        net = UNet_Univ5_Ablation(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_univ5_wo_uniprompt":
        net = UNet_UniV5_WO_Uni_Prompt(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size).cuda()
    elif net_type == "unet_univ5_attention_concat":
        net = UNet_UniV5_AttentionConcat(in_chns=in_chns, class_num=class_num, prompt_type=args.prompt, attention_type=args.attention,
                         sup_type=args.sup_type, use_label_prompt=args.label_prompt, client_num=args.min_num_clients, client_id=args.cid, img_size=args.img_size).cuda()
    
    else:
        net = None
    return net
