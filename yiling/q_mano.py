import os
import os.path as osp

import numpy as np
import torch
from torch.nn import Module

from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
from manopth import rodrigues_layer, rotproj, rot6d
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)

from nnutils import image_utils, geom_utils, mesh_utils, hand_utils

def eyen_like(n, T):
    return torch.eye(4)[None,None,:,:].expand(T.shape[0], T.shape[1],-1,-1)


class QManoLayer(Module):
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx', 'joint_rot_mode'
    ]

    def __init__(self,
                 center_idx=None,
                 flat_hand_mean=True,
                 ncomps=6,
                 side='right',
                 mano_root='mano/models',
                 use_pca=True,
                 root_rot_mode='axisang',
                 joint_rot_mode='axisang',
                 metric=1000.,
                 robust_rot=False):
        """
        Args:
            center_idx: index of center joint in our computations,
                if -1 centers on estimate of palm as middle of base
                of middle finger and wrist
            flat_hand_mean: if True, (0, 0, 0, ...) pose coefficients match
                flat hand, else match average hand pose
            mano_root: path to MANO pkl files for left and right hand
            ncomps: number of PCA components form pose space (<45)
            side: 'right' or 'left'
            use_pca: Use PCA decomposition for pose space.
            joint_rot_mode: 'axisang' or 'rotmat', ignored if use_pca
        """
        super().__init__()
        self.metric = metric
        self.center_idx = center_idx
        self.robust_rot = robust_rot
        if root_rot_mode == 'axisang':
            self.rot = 3
        else:
            self.rot = 6
        self.flat_hand_mean = flat_hand_mean
        self.side = side
        self.use_pca = use_pca
        self.joint_rot_mode = joint_rot_mode
        self.root_rot_mode = root_rot_mode
        if use_pca:
            self.ncomps = ncomps
        else:
            self.ncomps = 45

        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components']

        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]
                              ) if flat_hand_mean else smpl_data['hands_mean']
        hands_mean = hands_mean.copy()
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Save as axis-angle
            self.register_buffer('th_hands_mean', th_hands_mean)
            selected_components = hands_components[:ncomps]
            self.register_buffer('th_comps', torch.Tensor(hands_components))
            self.register_buffer('th_selected_comps',
                                 torch.Tensor(selected_components))
        else:
            th_hands_mean_rotmat = rodrigues_layer.batch_rodrigues(
                th_hands_mean.view(15, 3)).reshape(15, 3, 3)
            self.register_buffer('th_hands_mean_rotmat', th_hands_mean_rotmat)

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def forward(self,
                th_pose_coeffs,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1),
                share_betas=torch.Tensor([0]),
                ):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        """
        # if len(th_pose_coeffs) == 0:
        #     return th_pose_coeffs.new_empty(0), th_pose_coeffs.new_empty(0)

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Remove global rot coeffs
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                 self.ncomps]
            if self.use_pca:
                # PCA components --> axis angles
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            # Concatenate back global rot
            th_full_pose = torch.cat([
                th_pose_coeffs[:, :self.rot],
                self.th_hands_mean + th_full_hand_pose
            ], 1)
            if self.root_rot_mode == 'axisang':
                # compute rotation matrixes from axis-angle while skipping global rotation
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, (
                'When not self.use_pca, '
                'th_pose_coeffs should have 4 dims, got {}'.format(
                    th_pose_coeffs.dim()))
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_pose_coeffs.shape[2:4]))
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        # Full axis angle representation with root joint
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       self.th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)

        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        # th_v_posed = th_v_shaped + torch.matmul(
        #     self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        th_v_posed = th_v_shaped 
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results


        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)



        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]

        # print("th_verts_bkp", th_verts)
        # print("th_jtr", th_jtr)
        # print("LBS results", torch.matmul(self.th_weights, th_jtr.squeeze(0)))
        # exit()

        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        th_jtr = torch.cat([th_jtr, tips], 1)

        # # Reorder joints to match visualization utilities
        # th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        # Scale to milimeters
        # th_verts = th_verts * 1000 / self.metric
        # th_jtr = th_jtr * 1000 / self.metric
        return th_verts, th_jtr

    def compute_intermediate(self,
                th_pose_coeffs,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1),
                share_betas=torch.Tensor([0]),
                ):
        
        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Remove global rot coeffs
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                 self.ncomps]
            if self.use_pca:
                # PCA components --> axis angles
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            # Concatenate back global rot
            th_full_pose = torch.cat([
                th_pose_coeffs[:, :self.rot],
                self.th_hands_mean + th_full_hand_pose
            ], 1)
            if self.root_rot_mode == 'axisang':
                # compute rotation matrixes from axis-angle while skipping global rotation
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, (
                'When not self.use_pca, '
                'th_pose_coeffs should have 4 dims, got {}'.format(
                    th_pose_coeffs.dim()))
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_pose_coeffs.shape[2:4]))
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        # Full axis angle representation with root joint
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       self.th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)

        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        # th_v_posed = th_v_shaped + torch.matmul(
        #     self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        th_v_posed = th_v_shaped 
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results


        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        # tmp_vec = torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), joint_js.unsqueeze(3)], 3)
        G = eyen_like(4, th_results) - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), joint_js.unsqueeze(3)], 3)
        # print("G", G.shape)
        # print("th_results", th_results.shape)
        # th_results2 = torch.matmul(th_results, G).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))
        # print(th_T.shape, "th_T")

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)
        # print("th_rest_shape_h", th_rest_shape_h.shape)
        # print("th_T", th_T.shape)
        # print("(th_T * th_rest_shape_h.unsqueeze(1))", (th_T * th_rest_shape_h.unsqueeze(1)).shape)
        # exit()
        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]

        # print("th_verts_bkp", th_verts)
        # print("th_jtr", th_jtr)
        # print("LBS results", torch.matmul(self.th_weights, th_jtr.squeeze(0)))

        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        th_jtr = torch.cat([th_jtr, tips], 1)
        # tips_transform = torch.cat([tips, torch.zeros_like(tips)[:,:,0:1]], -1)[:,:,:, None]
        # tips_transform = torch.cat([torch.zeros(tips.shape[0], tips.shape[1], 4, 3), tips_transform], -1)
        # tips_transform = eyen_like(4, tips) + tips_transform
        # joint_transform = torch.cat([th_results_global, tips_transform], 1)
        # joint_transform = joint_transform[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        joint_transform = th_results_global
        # # Reorder joints to match visualization utilities
        # th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        if not bool(torch.norm(th_trans) == 0):
            joint_transform[:,:,:3,3] += th_trans.unsqueeze(1)
        # # Scale to milimeters
        # th_verts = th_verts * 1000 / self.metric
        # th_jtr = th_jtr * 1000 / self.metric
        return joint_transform.squeeze(0), G.squeeze(0), th_rest_shape_h.squeeze(0).transpose(1, 0)

    def forward_lbs(self, joint_transform, G, rest_shape, lbs_weights):
        th_results2 = torch.matmul(joint_transform, G).permute(1, 2, 0)
        th_T = torch.matmul(th_results2, lbs_weights.transpose(1, 0)).permute(2, 0, 1)
        # print("th_T.shape", th_T.shape)
        # print("rest_shape", rest_shape.shape)
        # print("th_results2", th_results2.shape)
        # th_T.shape torch.Size([778, 4, 4])
        # rest_shape torch.Size([778, 4])
        # th_results2 torch.Size([4, 4, 16])
        th_verts = (th_T * rest_shape.unsqueeze(1)).sum(2)
        th_verts = th_verts[:, :3]

        return th_verts

def get_parameters():
    pose = torch.tensor([0., 0.,  0.,  1.0677e-01,  4.5597e-02,
         5.8168e-01, -1.5491e-03,  4.6781e-19,  3.0077e-01, -8.7865e-04,
         1.0913e-18,  4.4659e-02,  4.8272e-02,  2.7133e-02,  6.8359e-01,
        -4.7846e-02, -1.4944e-17,  2.7921e-01, -2.3152e-02, -2.4992e-18,
         1.1335e-01, -1.9946e-01,  2.6292e-01,  5.5594e-01,  5.0050e-03,
        -4.0773e-03, -8.2433e-03,  7.8256e-02, -7.0344e-02, -1.4683e-01,
        -1.6839e-02, -4.1667e-03,  8.3948e-01,  9.1033e-02, -2.8175e-17,
        -2.8546e-01, -7.3624e-02,  1.4909e-17,  2.4677e-01,  6.5748e-01,
        -6.6345e-01,  6.6218e-01, -1.3192e-01,  3.5457e-01, -2.1654e-01,
         1.2122e-01, -2.9965e-01,  1.7670e-01])
    trans = torch.tensor([-0.1989,  0.0453, -0.3449]) 
    beta = torch.tensor([-0.8101, -0.7772, -1.9708, -0.3511, -0.6499, -2.7120,  1.1160, -0.6333,
         0.7500, -0.4506])
    return pose, trans, beta

def main():
    mano_path='externals/mano/'
    mano_layer_right = QManoLayer(
        mano_root=mano_path, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    pose, trans = torch.zeros([1, 48]), torch.zeros([1, 3])

    verts, joints = mano_layer_right(pose, th_trans=trans) # in MM
    faces = mano_layer_right.th_faces

    mesh_hand = mesh_utils.Meshes(verts, faces.unsqueeze(0))
    mesh_utils.dump_meshes(osp.join("outputs", 'test_hand_rest'), mesh_hand)


    joint_transform, G, rest_shape = mano_layer_right.compute_intermediate(pose, th_trans=trans)
    verts2 = mano_layer_right.forward_lbs(joint_transform, G, rest_shape, mano_layer_right.th_weights)

    # print("joint_transform", joint_transform)
    print(verts, "verts rest")
    print((verts - verts2).abs().max(), "verts rest")



    # new pose

    pose, trans, beta = get_parameters()
    pose, trans = pose.unsqueeze(0), trans.unsqueeze(0)

    trans = trans / 2

    verts, joints = mano_layer_right(pose, th_trans=trans)
    joint_transform, G_, rest_shape_ = mano_layer_right.compute_intermediate(pose, th_trans=trans)

    # print("joint_transform_new", joint_transform)
    verts2 = mano_layer_right.forward_lbs(joint_transform, G, rest_shape, mano_layer_right.th_weights)
    

    mesh_hand = mesh_utils.Meshes(verts2.unsqueeze(0), faces.unsqueeze(0))
    mesh_utils.dump_meshes(osp.join("outputs", 'test_hand_lbs'), mesh_hand)
    print(verts, "verts")
    print(verts2, "verts2")
    print((G_ - G).abs().max(), "G")
    print((rest_shape_ - rest_shape_).abs().max(), "rest_shape_")
    print((verts - verts2).abs().max(), "verts shaped")
    exit()
    verts = mano_layer_right.forward_given_js_T(js_T, trans)
    mesh_hand = mesh_utils.Meshes(verts, faces.unsqueeze(0))
    mesh_utils.dump_meshes(osp.join("outputs", 'test_hand_js'), mesh_hand)

    verts, joints = mano_layer_right(pose, th_trans=trans)
    mesh_hand = mesh_utils.Meshes(verts, faces.unsqueeze(0))
    mesh_utils.dump_meshes(osp.join("output", 'test_hand_deformed'), mesh_hand)





if __name__ == '__main__':
    main()





