import torch
import torch.nn.functional as F
from utils import losses, metrics, ramps
class AGEnergyLoss(torch.nn.Module):
    def forward(
            self, feature, sup_label, y_hat, kernels_desc, kernels_radius, sample, height_input, width_input,Unlabeled_RoIs,num_classes
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :return: Loss function value.
        """
        N, C, height_pred, width_pred = y_hat.shape
        _, _, height_feature, width_feature = feature.shape
        device = y_hat.device
        sup_label_downsample = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_feature, width_feature), mode='nearest')
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        prototype = AGEnergyLoss._prototype_generator(feature, sup_label_downsample, num_classes)
        print("prototype.shape = ", prototype.shape)
        prototype_l2 = F.normalize(prototype, p=2, dim=2)#num_classes,N,C
        feature_upsample = F.interpolate(feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        affinity_map = AGEnergyLoss._affinity(feature_upsample, prototype_l2, num_classes)# N,num_class,H,W
        # ## redefined prediction(Affinity)
        y_redefined = affinity_map * y_hat
        y_hat_softmax=torch.softmax(y_redefined, dim=1)
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        
        ###
        

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
            width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )
        # print(kernels.shape)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))

        # denom = N * height_pred * width_pred
        denom = Unlabeled_RoIs.sum()

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))
        # print(kernels.shape)
        # print(y_hat_unfolded.shape)
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)

        print(product_kernel_x_y_hat.shape)
        print(y_hat_softmax.shape)
        print(Unlabeled_RoIs.shape)
        # Using shortcut for Pott's class compatibility model
        loss = -(product_kernel_x_y_hat * y_hat_softmax * Unlabeled_RoIs).sum()
        # comment out to save computation, total loss may go below 0
        loss = kernels.sum() + loss
        loss = torch.clamp(loss, min=1e-5)


        out = {
            'loss': loss / denom, 'prediction_redefined':y_redefined, 'heated_map':product_kernel_x_y_hat
        }

        return out

    @staticmethod
    def _downsample(img, height_dst, width_dst):
        f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = AGEnergyLoss._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = AGEnergyLoss._downsample(
                        feature, height_pred, width_pred
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                AGEnergyLoss._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = AGEnergyLoss._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(
                1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(
                1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)#N,1,H,W
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, N, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = torch.sum(weight, dim=[2,3], keepdim=False)

        print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)/labeled_num
        return prototype
            
    @staticmethod
    def _affinity(feature, pro_vector, num_classes):
        # print("pro_vector.shape = ",pro_vector.shape)
        N, C, H, W = feature.shape
        # Reshape feature to (N, H, W, C)&pro_vector to (num_class, N, C)
        feature = feature.transpose(1,3).transpose(1,2)
        feature = F.normalize(feature, p=2, dim=3)#N,H,W,C
        # pro_vector = pro_vector.view(N, C, num_classes).transpose(0,2).transpose(1,2)
        # print("pro_vector.shape = ",pro_vector.shape)
        # print(feature.shape)
        cosine_similarities_prototype = torch.zeros((N, H, W, num_classes)).to(feature.device)
 
        
        for c in range(num_classes):
            pix_prototype = pro_vector[c,:,:].unsqueeze(1).unsqueeze(1)#N,1,1,C
            # pix_prototype = F.normalize(pix_prototype, p=2, dim=-1)
            # print("pix_prototype.shape=",pix_prototype.shape)
            cosine_similarities_prototype[:,:,:,c] = F.cosine_similarity(feature,pix_prototype,dim=-1)
        cosine_similarities_prototype[cosine_similarities_prototype<0] = 0

        ###just max num_classes
        # max_values, _ = torch.max(cosine_similarities_prototype, dim=-1, keepdim=True)
        # result = torch.zeros_like(cosine_similarities_prototype).to(feature.device)
        # result[max_values == cosine_similarities_prototype] = cosine_similarities_prototype[max_values == cosine_similarities_prototype]
        ###

        affinity_map = cosine_similarities_prototype/torch.clamp(torch.sum(cosine_similarities_prototype, dim=-1, keepdim=True), min=1e-10)
        affinity_map = affinity_map.transpose(1,3).transpose(2,3)
        # print("affinity.shape = ", affinity_map.shape)
        return affinity_map# N,num_class,H,W




class AGEnergyLoss_PixelsAssignment(torch.nn.Module):
    def forward(
            self, feature, sup_label, y_hat, kernels_desc, kernels_radius, sample, height_input, width_input,Unlabeled_RoIs,Unlabeled_RoIs_sup,num_classes
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :return: Loss function value.
        """

        # refined_prediction
        N, C, height_pred, width_pred = y_hat.shape
        _, _, height_feature, width_feature = feature.shape
        device = y_hat.device
        sup_label_downsample = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_feature, width_feature), mode='nearest')
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        prototype = AGEnergyLoss_PixelsAssignment._prototype_generator(feature, sup_label_downsample, num_classes)
        print("prototype.shape = ", prototype.shape)
        prototype_l2 = F.normalize(prototype, p=2, dim=2)#num_classes,N,C
        feature_upsample = F.interpolate(feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        affinity_map = AGEnergyLoss_PixelsAssignment._affinity(feature_upsample, prototype_l2, num_classes)# N,num_class,H,W
        # ## redefined prediction(Affinity)
        y_redefined = affinity_map * y_hat
        y_hat_softmax=torch.softmax(y_redefined, dim=1)
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        ###

        # ## bounder
        
        bounder_loss = AGEnergyLoss_PixelsAssignment._bounder_aware_loss(y_redefined, y_hat, Unlabeled_RoIs)
        

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
            width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )
        # kernels = kernels * (Unlabeled_RoIs_sup.unsqueeze(2).unsqueeze(2))

        # denom = N * height_pred * width_pred
        denom = Unlabeled_RoIs_sup.sum()

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))
        # print(kernels.shape)
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)
        


        # Using shortcut for Pott's class compatibility model
        loss = -(product_kernel_x_y_hat * y_hat_softmax * Unlabeled_RoIs_sup).sum()
        # comment out to save computation, total loss may go below 0
        loss = kernels.sum() + loss
        loss = torch.clamp(loss, min=1e-5)


        out = {
            'loss': loss / denom,'bounder_loss':bounder_loss, 'prediction_redefined':y_redefined, 'heated_map':product_kernel_x_y_hat
        }

        return out

    @staticmethod
    def _downsample(img, height_dst, width_dst):
        f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = AGEnergyLoss_PixelsAssignment._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = AGEnergyLoss_PixelsAssignment._downsample(
                        feature, height_pred, width_pred
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                AGEnergyLoss_PixelsAssignment._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = AGEnergyLoss_PixelsAssignment._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(
                1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(
                1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)#N,1,H,W
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, N, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = torch.sum(weight, dim=[2,3], keepdim=False)

        print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1e-4)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)/labeled_num
        return prototype
            
    @staticmethod
    def _affinity(feature, pro_vector, num_classes):
        # print("pro_vector.shape = ",pro_vector.shape)
        N, C, H, W = feature.shape
        # Reshape feature to (N, H, W, C)&pro_vector to (num_class, N, C)
        feature = feature.transpose(1,3).transpose(1,2)
        feature = F.normalize(feature, p=2, dim=3)#N,H,W,C
        # pro_vector = pro_vector.view(N, C, num_classes).transpose(0,2).transpose(1,2)
        # print("pro_vector.shape = ",pro_vector.shape)
        # print(feature.shape)
        cosine_similarities_prototype = torch.zeros((N, H, W, num_classes)).to(feature.device)
 
        
        for c in range(num_classes):
            pix_prototype = pro_vector[c,:,:].unsqueeze(1).unsqueeze(1)#N,1,1,C
            # pix_prototype = F.normalize(pix_prototype, p=2, dim=-1)
            # print("pix_prototype.shape=",pix_prototype.shape)
            cosine_similarities_prototype[:,:,:,c] = F.cosine_similarity(feature,pix_prototype,dim=-1)
        cosine_similarities_prototype[cosine_similarities_prototype<0] = 0

        ###just max num_classes
        # max_values, _ = torch.max(cosine_similarities_prototype, dim=-1, keepdim=True)
        # result = torch.zeros_like(cosine_similarities_prototype).to(feature.device)
        # result[max_values == cosine_similarities_prototype] = cosine_similarities_prototype[max_values == cosine_similarities_prototype]
        ###

        affinity_map = cosine_similarities_prototype/torch.clamp(torch.sum(cosine_similarities_prototype, dim=-1, keepdim=True), min=1e-10)
        affinity_map = affinity_map.transpose(1,3).transpose(2,3)
        # print("affinity.shape = ", affinity_map.shape)
        return affinity_map# N,num_class,H,W
    @staticmethod
    def _bounder_aware_loss(soft_label, y_hat, Unlabeled_RoIs):
        y_hat_softmax=torch.softmax(y_hat, dim=1)
        soft_label = soft_label.transpose(1,3).transpose(1,2)# N,H,W,num_class
        # ##max
        max_values, _ = torch.max(soft_label, dim=-1, keepdim=True)
        result = torch.zeros_like(soft_label).to(y_hat.device)
        result[max_values == soft_label] = soft_label[max_values == soft_label]
        soft_label = torch.softmax(result, dim=-1)

        # soft_label = torch.softmax(soft_label, dim=-1)
        soft_label = soft_label.transpose(1,3).transpose(2,3)# N,num_class,H,W
        # soft_label = soft_label.transpose(1,3).transpose(2,3)# N,num_class,H,W
        soft_label_bounder = soft_label * Unlabeled_RoIs
        y_hat_softmax_bounder = y_hat_softmax * Unlabeled_RoIs
        return losses.dice_loss1(y_hat_softmax_bounder, soft_label_bounder)



class AGEnergyLoss_EnergyMap_PixelsAssignment(torch.nn.Module):
    def forward(
            self, feature, sup_label, y_hat, kernels_desc, kernels_radius, sample, height_input, width_input,Unlabeled_RoIs,Unlabeled_RoIs_sup,num_classes
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :return: Loss function value.
        """

        # refined_prediction
        N, C, height_pred, width_pred = y_hat.shape
        _, _, height_feature, width_feature = feature.shape
        device = y_hat.device
        sup_label_downsample = F.interpolate(sup_label.unsqueeze(1).float(), size=(height_feature, width_feature), mode='nearest')
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        prototype = AGEnergyLoss_PixelsAssignment._prototype_generator(feature, sup_label_downsample, num_classes)
        print("prototype.shape = ", prototype.shape)
        prototype_l2 = F.normalize(prototype, p=2, dim=2)#num_classes,N,C
        feature_upsample = F.interpolate(feature, size=(height_pred, width_pred), mode='bilinear', align_corners=False)
        print("sup_label_downsample.shape = ", sup_label_downsample.shape)
        affinity_map = AGEnergyLoss_PixelsAssignment._affinity(feature_upsample, prototype_l2, num_classes)# N,num_class,H,W
        # ## redefined prediction(Affinity)
        y_redefined = affinity_map * y_hat
        y_hat_softmax=torch.softmax(y_redefined, dim=1)
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        ###
        
        
        

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
            width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )
        # kernels = kernels * (Unlabeled_RoIs_sup.unsqueeze(2).unsqueeze(2))

        # denom = N * height_pred * width_pred
        denom = Unlabeled_RoIs_sup.sum()

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        # kernels = kernels * (Unlabeled_RoIs.unsqueeze(2).unsqueeze(2))
        # print(kernels.shape)
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred) \
            .sum(dim=2, keepdim=False)

        energy_field = product_kernel_x_y_hat * y_hat_softmax
        # Using shortcut for Pott's class compatibility model
        bounder_loss = AGEnergyLoss_EnergyMap_PixelsAssignment._bounder_aware_loss(energy_field, y_hat, Unlabeled_RoIs)
        loss = -(energy_field * Unlabeled_RoIs_sup).sum()
        # comment out to save computation, total loss may go below 0
        loss = kernels.sum() + loss
        loss = torch.clamp(loss, min=1e-5)


        out = {
            'loss': loss / denom,'bounder_loss':bounder_loss, 'prediction_redefined':y_redefined, 'heated_map':product_kernel_x_y_hat
        }

        return out

    @staticmethod
    def _downsample(img, height_dst, width_dst):
        f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = AGEnergyLoss_EnergyMap_PixelsAssignment._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = AGEnergyLoss_EnergyMap_PixelsAssignment._downsample(
                        feature, height_pred, width_pred
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                AGEnergyLoss_EnergyMap_PixelsAssignment._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = AGEnergyLoss_EnergyMap_PixelsAssignment._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius,
                                    radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(
                1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(
                1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _prototype_generator(feature, sup_label, num_classes):
        N, C, H, W = feature.shape #N,C,H,W
        if len(sup_label.shape) != 4:
            sup_label = sup_label.unsqueeze(1)#N,1,H,W
        sup_label = sup_label.long()
        labeled_num = torch.zeros((num_classes, N, 1)).to(feature.device)
        
        prototype = torch.zeros((num_classes, N, C, H, W)).to(feature.device)
        # print("prototype.shape = ",prototype.shape)
        for num in range(num_classes):
            weight = torch.zeros_like(sup_label).to(feature.device)
            weight[sup_label==num]=1
            prototype[num][:] = feature * weight
            labeled_num[num] = torch.sum(weight, dim=[2,3], keepdim=False)

        print("prototype.shape = ",prototype.shape)
        # # Unify prototype num N C
        labeled_num = torch.clamp(labeled_num, min=1e-4)
        prototype = torch.sum(prototype, dim=[3,4], keepdim=False)/labeled_num
        return prototype
            
    @staticmethod
    def _affinity(feature, pro_vector, num_classes):
        # print("pro_vector.shape = ",pro_vector.shape)
        N, C, H, W = feature.shape
        # Reshape feature to (N, H, W, C)&pro_vector to (num_class, N, C)
        feature = feature.transpose(1,3).transpose(1,2)
        feature = F.normalize(feature, p=2, dim=3)#N,H,W,C
        # pro_vector = pro_vector.view(N, C, num_classes).transpose(0,2).transpose(1,2)
        # print("pro_vector.shape = ",pro_vector.shape)
        # print(feature.shape)
        cosine_similarities_prototype = torch.zeros((N, H, W, num_classes)).to(feature.device)
 
        
        for c in range(num_classes):
            pix_prototype = pro_vector[c,:,:].unsqueeze(1).unsqueeze(1)#N,1,1,C
            # pix_prototype = F.normalize(pix_prototype, p=2, dim=-1)
            # print("pix_prototype.shape=",pix_prototype.shape)
            cosine_similarities_prototype[:,:,:,c] = F.cosine_similarity(feature,pix_prototype,dim=-1)
        cosine_similarities_prototype[cosine_similarities_prototype<0] = 0

        ###just max num_classes
        # max_values, _ = torch.max(cosine_similarities_prototype, dim=-1, keepdim=True)
        # result = torch.zeros_like(cosine_similarities_prototype).to(feature.device)
        # result[max_values == cosine_similarities_prototype] = cosine_similarities_prototype[max_values == cosine_similarities_prototype]
        ###

        # affinity_map = cosine_similarities_prototype.exp().transpose(1,3).transpose(2,3)
        affinity_map = cosine_similarities_prototype/torch.clamp(torch.sum(cosine_similarities_prototype, dim=-1, keepdim=True), min=1e-10)
        affinity_map = affinity_map.transpose(1,3).transpose(2,3)
        # print("affinity.shape = ", affinity_map.shape)
        return affinity_map# N,num_class,H,W
    @staticmethod
    def _bounder_aware_loss(energy_field, y_hat, Unlabeled_RoIs):
        y_hat_softmax=torch.softmax(y_hat, dim=1)
        energy_field = energy_field.transpose(1,3).transpose(1,2)# N,H,W,num_class
        # ##max
        # max_values, _ = torch.max(energy_field, dim=-1, keepdim=True)
        # result = torch.zeros_like(energy_field).to(y_hat.device)
        # result[max_values == energy_field] = energy_field[max_values == energy_field]
        # energy_field = torch.softmax(result, dim=-1)

        energy_field = torch.softmax(energy_field, dim=-1)
        energy_field = energy_field.transpose(1,3).transpose(2,3)# N,num_class,H,W
        # energy_field = energy_field.transpose(1,3).transpose(2,3)# N,num_class,H,W
        soft_label_bounder = energy_field * Unlabeled_RoIs
        y_hat_softmax_bounder = y_hat_softmax * Unlabeled_RoIs
        return losses.dice_loss1(y_hat_softmax_bounder, soft_label_bounder)






