# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from warp.sim.model import JOINT_COMPOUND, JOINT_REVOLUTE, JOINT_UNIVERSAL
from warp.utils import transform_identity

import math
import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp
import warp.sim 

from warp.utils import quat_to_matrix
import torch

def parse_lbs(builder, lbs_weights, lbs_transforms, lbs_G, lbs_rest, lbs_base_transform, lbs_scale, lbs_verts, lbs_faces):
    builder.add_lbs(lbs_weights, lbs_transforms, lbs_G, lbs_rest, lbs_base_transform, lbs_scale, lbs_verts, lbs_faces)

    # joint_transform = torch.tensor(lbs_transforms)
    # G = torch.tensor(lbs_G)
    # rest_shape = torch.tensor(lbs_rest)
    # lbs_weights = torch.tensor(lbs_weights)
    # th_results2 = torch.matmul(joint_transform, G).permute(1, 2, 0)
    # th_T = torch.matmul(th_results2, lbs_weights.transpose(1, 0)).permute(2, 0, 1)
    # th_verts = (th_T * rest_shape.unsqueeze(1)).sum(2)
    # th_verts = th_verts.cpu().detach().numpy()[:,:3]

    # print((lbs_verts - th_verts).max())
    # exit()


def update_lbs(model, body_q):

    joint_transform = model.lbs_transforms.numpy()
    G = model.lbs_G.numpy()
    rest_shape = model.lbs_rest.numpy()
    lbs_weights = model.lbs_weights.numpy()

    # print("joint_transform", joint_transform.shape)
    # print("G", G.shape)
    # print("rest_shape", rest_shape.shape)
    # print("lbs_weights", lbs_weights.shape)

    def transform_to_matrix(q):
        mat = np.eye(4)
        mat[:3, 3] = q[:3] / model.lbs_scale
        mat[:3,:3] = quat_to_matrix(q[3:])
        # print(quat_to_matrix(q[3:]))
        return mat

    # print(joint_transform[5:10], "joint_transform", joint_transform.shape)
    # print(body_q.numpy()[:5], "body_q")
    joint_transform = np.array([transform_to_matrix(q) for q in body_q.numpy()])
    # print(joint_transform[5:10], "joint_transform new")
    # print(np.abs(joint_transform-joint_transform_old), "diff")
    # exit()
    joint_transform = torch.tensor(joint_transform).float()
    G = torch.tensor(G)
    rest_shape = torch.tensor(rest_shape)
    lbs_weights = torch.tensor(lbs_weights)
    th_results2 = torch.matmul(joint_transform, G).permute(1, 2, 0)
    th_T = torch.matmul(th_results2, lbs_weights.transpose(1, 0)).permute(2, 0, 1)
    th_verts = (th_T * rest_shape.unsqueeze(1)).sum(2)
    th_verts = th_verts.cpu().detach().numpy()[:, :3] * model.lbs_scale

    # joint_transform = np.array([transform_to_matrix(q) for q in body_q.numpy()])
    # th_results2 = np.matmul(joint_transform, G).transpose(1, 2, 0)
    # th_T = np.matmul(th_results2, lbs_weights.transpose(1, 0)).transpose(2, 0, 1)
    # th_verts = (th_T * np.expand_dims(rest_shape, 1)).sum(2)

    # th_verts = th_verts[:, :3] * 20

    # old_verts = model.lbs_verts.numpy()
    # print((old_verts - th_verts).max())
    # exit()
    model.lbs_verts = wp.array(th_verts, dtype=wp.vec3)
    

    # wp.launch(kernel=lbs_kernel_update,
    #             dim=model.articulation_count,
    #             inputs=[    
    #             body_q,
    #             model.lbs_weights,
    #             model.lbs_transforms,
    #             model.lbs_G,
    #             model.lbs_rest,
    #             ],
    #             outputs=[
    #             model.lbs_verts,
    #             ],
    #             device=model.device)


    #             m.lbs_verts_count = len(self.lbs_verts[0])
    #             m.lbs_faces = wp.array(self.lbs_faces[0], dtype=wp.int32)
    #             m.lbs_verts = wp.array(self.lbs_verts[0], dtype=wp.vec3)

    #             m.lbs_weights = wp.array(self.lbs_weights[0], dtype=wp.float32)
    #             m.lbs_transforms = wp.array(self.lbs_transforms[0], dtype=wp.mat44)
    #             m.lbs_G = wp.array(self.lbs_G[0], dtype=wp.mat44)
    #             m.lbs_rest = wp.array(self.lbs_rest[0], dtype=wp.vec4)

# def update_lbs(model, body_q):

#     wp.launch(kernel=lbs_kernel_update,
#                 dim=model.articulation_count,
#                 inputs=[    
#                 body_q,
#                 model.lbs_weights,
#                 model.lbs_transforms,
#                 model.lbs_G,
#                 model.lbs_rest,
#                 ],
#                 outputs=[
#                 model.lbs_verts,
#                 ],
#                 device=model.device)


#                 m.lbs_verts_count = len(self.lbs_verts[0])
#                 m.lbs_faces = wp.array(self.lbs_faces[0], dtype=wp.int32)
#                 m.lbs_verts = wp.array(self.lbs_verts[0], dtype=wp.vec3)

#                 m.lbs_weights = wp.array(self.lbs_weights[0], dtype=wp.float32)
#                 m.lbs_transforms = wp.array(self.lbs_transforms[0], dtype=wp.mat44)
#                 m.lbs_G = wp.array(self.lbs_G[0], dtype=wp.mat44)
#                 m.lbs_rest = wp.array(self.lbs_rest[0], dtype=wp.vec4)

# @wp.kernel
# def lbs_kernel_update(
#                 body_q:
#                 model.lbs_weights,
#                 model.lbs_transforms,
#                 model.lbs_G,
#                 model.lbs_rest,
#     body_q: wp.array(dtype=wp.transform),
#     articulation_mask: wp.array(dtype=int), # used to enable / disable FK for an articulation, if None then treat all as enabled
#     joint_q: wp.array(dtype=float),
#     joint_qd: wp.array(dtype=float),
#     joint_q_start: wp.array(dtype=int),
#     joint_qd_start: wp.array(dtype=int),
#     joint_type: wp.array(dtype=int),
#     joint_parent: wp.array(dtype=int),
#     joint_X_p: wp.array(dtype=wp.transform),
#     joint_X_c: wp.array(dtype=wp.transform),
#     joint_axis: wp.array(dtype=wp.vec3),
#     body_com: wp.array(dtype=wp.vec3),
#     # outputs
#     body_q: wp.array(dtype=wp.transform),
#     body_qd: wp.array(dtype=wp.spatial_vector)):

#     tid = wp.tid()

#     # early out if disabling FK for this articulation
#     if (articulation_mask):
#         if (articulation_mask[tid]==0):
#             return

#     joint_start = articulation_start[tid]
#     joint_end = articulation_start[tid+1]

#     for i in range(joint_start, joint_end):

#         parent = joint_parent[i]
#         X_wp = wp.transform_identity()
#         v_wp = wp.spatial_vector()

#         if (parent >= 0):
#             X_wp = body_q[parent]
#             v_wp = body_qd[parent]

#         # compute transform across the joint
#         type = joint_type[i]
#         axis = joint_axis[i]

#         X_pj = joint_X_p[i]
#         X_cj = joint_X_c[i]  
        
#         q_start = joint_q_start[i]
#         qd_start = joint_qd_start[i]

#         if type == wp.sim.JOINT_PRISMATIC:

#             q = joint_q[q_start]
#             qd = joint_qd[qd_start]

#             X_jc = wp.transform(axis*q, wp.quat_identity())
#             v_jc = wp.spatial_vector(wp.vec3(), axis*qd)

#         if type == wp.sim.JOINT_REVOLUTE:

#             q = joint_q[q_start]
#             qd = joint_qd[qd_start]

#             X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
#             v_jc = wp.spatial_vector(axis*qd, wp.vec3())

#         if type == wp.sim.JOINT_BALL:

#             r = wp.quat(joint_q[q_start+0],
#                         joint_q[q_start+1],
#                         joint_q[q_start+2],
#                         joint_q[q_start+3])

#             w = wp.vec3(joint_qd[qd_start+0],
#                         joint_qd[qd_start+1],
#                         joint_qd[qd_start+2])

#             # print(r)
            
#             X_jc = wp.transform(wp.vec3(), r)
#             v_jc = wp.spatial_vector(w, wp.vec3())

#         if type == wp.sim.JOINT_FIXED:
            
#             X_jc = wp.transform_identity()
#             v_jc = wp.spatial_vector(wp.vec3(), wp.vec3())

#         if type == wp.sim.JOINT_FREE:

#             t = wp.transform(
#                     wp.vec3(joint_q[q_start+0], joint_q[q_start+1], joint_q[q_start+2]),
#                     wp.quat(joint_q[q_start+3], joint_q[q_start+4], joint_q[q_start+5], joint_q[q_start+6]))

#             v = wp.spatial_vector(
#                     wp.vec3(joint_qd[qd_start+0], joint_qd[qd_start+1], joint_qd[qd_start+2]),
#                     wp.vec3(joint_qd[qd_start+3], joint_qd[qd_start+4], joint_qd[qd_start+5]))

#             X_jc = t
#             v_jc = v

#         if type == wp.sim.JOINT_COMPOUND:

#             q_c = wp.transform_get_rotation(X_cj)

#             # body local axes
#             local_0 = wp.quat_rotate(q_c, wp.vec3(1.0, 0.0, 0.0))
#             local_1 = wp.quat_rotate(q_c, wp.vec3(0.0, 1.0, 0.0))
#             local_2 = wp.quat_rotate(q_c, wp.vec3(0.0, 0.0, 1.0))

#             # reconstruct rotation axes, todo: can probably use fact that rz'*ry'*rx' == rx*ry*rz to avoid some work here
#             axis_0 = local_0
#             q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start+0])

#             axis_1 = wp.quat_rotate(q_0, local_1)
#             q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start+1])

#             axis_2 = wp.quat_rotate(q_1*q_0, local_2)
#             q_2 = wp.quat_from_axis_angle(axis_2, joint_q[q_start+2])

#             t = wp.transform(wp.vec3(), q_2*q_1*q_0)

#             v = wp.spatial_vector(axis_0*joint_qd[qd_start+0] + 
#                                   axis_1*joint_qd[qd_start+1] + 
#                                   axis_2*joint_qd[qd_start+2], wp.vec3())

#             X_jc = t
#             v_jc = v

#         if type == wp.sim.JOINT_UNIVERSAL:

#             q_c = wp.transform_get_rotation(X_cj)

#             # body local axes
#             local_0 = wp.quat_rotate(q_c, wp.vec3(1.0, 0.0, 0.0))
#             local_1 = wp.quat_rotate(q_c, wp.vec3(0.0, 1.0, 0.0))

#             # reconstruct rotation axes
#             axis_0 = local_0
#             q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start+0])

#             axis_1 = wp.quat_rotate(q_0, local_1)
#             q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start+1])

#             t = wp.transform(wp.vec3(), q_1*q_0)

#             v = wp.spatial_vector(axis_0*joint_qd[qd_start+0] + 
#                                   axis_1*joint_qd[qd_start+1], wp.vec3())

#             X_jc = t
#             v_jc = v


#         X_wj = X_wp*X_pj
#         X_wc = X_wj*X_jc

#         # transform velocity across the joint to world space
#         angular_vel = wp.transform_vector(X_wj, wp.spatial_top(v_jc))
#         linear_vel = wp.transform_vector(X_wj, wp.spatial_bottom(v_jc))

#         v_wc = v_wp + wp.spatial_vector(angular_vel, linear_vel + wp.cross(angular_vel, body_com[i]))

#         body_q[i] = X_wc
#         body_qd[i] = v_wc
#         