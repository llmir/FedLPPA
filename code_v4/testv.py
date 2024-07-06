from networks.unet import UNet, UNet_LC

net = UNet_LC(in_chns=3, class_num=2, pcs_num=1, emb_num=5, client_num=5, client_id=0)

i = 0
for name, param in net.named_parameters():
    print(name)
    if 'decoder' in name:
        i = i + 1

print(i)


i = 0
for name, param in net.named_parameters():
    if 'up3' in name or 'up4' in name or 'out_conv' in name:
        i = i + 1

print(i)
