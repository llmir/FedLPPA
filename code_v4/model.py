from networks.unet import UNet_LC_MultiHead,UNet,UNet_UniV2,UNet_UniV3,UNet_UniV4,UNet_UniV5

model = UNet_UniV5(in_chns=3, class_num=3, prompt_type='universal', attention_type='dual',
                         sup_type='scribble', use_label_prompt=1, client_num=4, client_id=0, img_size=256).cuda()
i=0
# p_keywords = ['decoder.up3']
for name, param in model.named_parameters():
    # if p_keywords[0] in name:
    print("name = ",name)
        # print(name)
        # print("weight = ",param)
        # print(param.shape)
    i=i+1
print(i)
