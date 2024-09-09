from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.utils.merge import get_qkv
from lib.models.stark import build_stark_lightning_x_trt
from lib.test.tracker.stark_utils import PreprocessorX
from lib.utils.box_ops import clip_box
from lib.models.stark.repvgg import repvgg_model_convert
# for onnxruntime
from lib.test.tracker.stark_utils import PreprocessorX_onnx
import onnxruntime
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class STARK_LightningXtrt(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_LightningXtrt, self).__init__(params)
        network = build_stark_lightning_x_trt(params.cfg, phase='test')
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        repvgg_model_convert(network)
        network.deep_sup = False  # disable deep supervision during the test stage
        network.distill = False  # disable distillation during the test stage
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorX()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template, template_mask = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template, zx="template0", mask=template_mask)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)


                # Assuming feat has the shape [64, 1, 128], reshape it to [64, 128] for t-SNE
        feat = self.z_dict1['feat'].cpu() # Replace with actual embeddings
        feat_reshaped = feat.squeeze(1)  # Shape becomes [64, 128]

        # Apply t-SNE to reduce the dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        feat_tsne = tsne.fit_transform(feat_reshaped)  # Shape [64, 2]

        # Plot the result
        plt.figure(figsize=(8, 6))
        plt.scatter(feat_tsne[:, 0], feat_tsne[:, 1], cmap='viridis')
        plt.title("t-SNE Visualization of Feature Embeddings PyTorch")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar()
        plt.show()


        # Reshape pos to [64, 128] and apply t-SNE
        pos = self.z_dict1['pos'].cpu()  # Replace with actual embeddings
        pos_reshaped = pos.squeeze(1)  # Shape becomes [64, 128]

        pos_tsne = tsne.fit_transform(pos_reshaped)  # Shape [64, 2]

        # Plot positional embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(pos_tsne[:, 0], pos_tsne[:, 1], cmap='plasma')
        plt.title("t-SNE Visualization of Positional Embeddings PyTorch")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar()
        plt.show()

        with torch.no_grad():
            x_dict = self.network.forward_backbone(search, zx="search", mask=search_mask)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            q, k, v, key_padding_mask = get_qkv(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(q=q, k=k, v=v, key_padding_mask=key_padding_mask)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        # if self.debug:
        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        # cv2.imwrite(save_path, image_BGR)
        cv2.imshow('', image_BGR)
        cv2.waitKey(1)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


class STARK_LightningXtrt_onnx(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_LightningXtrt_onnx, self).__init__(params)
        """build two sessions"""
        '''2021.7.5 Add multiple gpu support'''
        # print("gpu_id", gpu_id)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ort_sess_z = onnxruntime.InferenceSession("backbone_bottleneck_pe.onnx", providers=providers)
        self.ort_sess_x = onnxruntime.InferenceSession("complete.onnx", providers=providers)
        self.preprocessor = PreprocessorX_onnx()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.ort_outs_z = []

    def initialize(self, image, info: dict):

        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template, template_mask = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # forward the template once
        ort_inputs = {'img_z': template, 'mask_z': template_mask}
        self.ort_outs_z = self.ort_sess_z.run(None, ort_inputs)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)


        # Assuming feat has the shape [64, 1, 128], reshape it to [64, 128] for t-SNE
        feat = self.ort_outs_z[0]  # Replace with actual embeddings
        feat_reshaped = feat.squeeze(1)  # Shape becomes [64, 128]

        # Apply t-SNE to reduce the dimensionality
        tsne = TSNE(n_components=2, random_state=42)
        feat_tsne = tsne.fit_transform(feat_reshaped)  # Shape [64, 2]

        # Plot the result
        plt.figure(figsize=(8, 6))
        plt.scatter(feat_tsne[:, 0], feat_tsne[:, 1], cmap='viridis')
        plt.title("t-SNE Visualization of Feature Embeddings ONNX")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar()
        plt.show()


        # Reshape pos to [64, 128] and apply t-SNE
        pos = self.ort_outs_z[2]  # Replace with actual embeddings
        pos_reshaped = pos.squeeze(1)  # Shape becomes [64, 128]

        pos_tsne = tsne.fit_transform(pos_reshaped)  # Shape [64, 2]

        # Plot positional embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(pos_tsne[:, 0], pos_tsne[:, 1], cmap='plasma')
        plt.title("t-SNE Visualization of Positional Embeddings ONNX")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar()
        plt.show()


        ort_inputs = {'img_x': search,
                      'mask_x': search_mask,
                      'feat_vec_z': self.ort_outs_z[0],
                      'mask_vec_z': self.ort_outs_z[1],
                      'pos_vec_z': self.ort_outs_z[2],
                      }

        ort_outs = self.ort_sess_x.run(None, ort_inputs)

        pred_box = (ort_outs[0].reshape(4) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        # if self.debug:
        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        cv2.imshow('', image_BGR)
        cv2.waitKey(1)
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def get_tracker_class():
    use_onnx = False
    if use_onnx:
        print("Using onnx model")
        return STARK_LightningXtrt_onnx
    else:
        print("Using original pytorch model")
        return STARK_LightningXtrt
