# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Humanoid
#
# Shows how to set up a simulation of a rigid-body Humanoid articulation based
# on the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################


import os
import math

import numpy as np

from pytorch3d.io import load_obj

import warp as wp
import warp.sim
import warp.sim.render

from q_mano import QManoLayer
import torch

# from warp.sim.import_lbs import parse_lbs

wp.init()

def get_lbs_variables():

    verts, faces, aux = load_obj('test_hand_rest_0.obj')
    # verts, faces, aux = load_obj('hand_fill2.obj')
    vertices, indices = verts.cpu().detach().numpy(), faces.verts_idx.cpu().detach().numpy()

    mano_path='externals/mano/'
    mano_layer_right = QManoLayer(
        mano_root=mano_path, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    pose, trans = torch.zeros([1, 48]), torch.zeros([1, 3])
    joint_transform, G_, rest_shape_ = mano_layer_right.compute_intermediate(pose, th_trans=trans)
    mano_path='externals/mano/'
    mano_layer_right = QManoLayer(
        mano_root=mano_path, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    pose, trans = torch.zeros([1, 48]), torch.zeros([1, 3])
    joint_transform, G, rest_shape = mano_layer_right.compute_intermediate(pose, th_trans=trans)

    lbs_weights = mano_layer_right.th_weights
    scale = 20
    base_transform = wp.transform(joint_transform[0,:3,3] * scale, (0.0, 0.0, 0.0, 1.0))
    return lbs_weights.cpu().detach().numpy(), joint_transform.cpu().detach().numpy(), \
        G.cpu().detach().numpy(), rest_shape.cpu().detach().numpy(), base_transform, \
        scale, vertices, indices

class Robot:

    frame_dt = 1.0/ (60.0)
    # frame_dt = 1.0/ (60.0)

    episode_duration = 2.6  # 5.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, render=True, num_envs=1, device=None):

        builder = wp.sim.ModelBuilder()
        articulation_builder = wp.sim.ModelBuilder()

        self.render = render

        self.num_envs = num_envs

        self.points_a = []
        self.points_b = []
        # number of contact points to visualize
        self.max_contact_count = 0

        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "test2.xml"),
            # os.path.join(os.path.dirname(__file__), "assets/nv_humanoid.xml"),
            articulation_builder,
            stiffness=0.0,
            damping=0.1,
            armature=0.007,
            armature_scale=10.0,
            contact_ke=1.e+4,
            contact_kd=1.e+2,
            contact_kf=1.e+2,
            contact_mu=0.5,
            contact_restitution=0,
            limit_ke=1.e+2,
            limit_kd=1.e+1)

        lbs_weights, lbs_transforms, lbs_G, lbs_rest, lbs_base_transform, lbs_scale, lbs_verts, lbs_faces = get_lbs_variables()
        # print("lbs_weights", lbs_weights.shape)
        # print("lbs_transforms", lbs_transforms.shape)
        # print("lbs_G", lbs_G.shape)
        # print("lbs_rest", lbs_rest.shape)
        # print("lbs_verts", lbs_verts.shape)
        # print("lbs_faces", lbs_faces.shape)
        # lbs_weights (778, 16)
        # lbs_transforms (16, 4, 4)
        # lbs_G (16, 4, 4)
        # lbs_rest (778, 4)
        # lbs_verts (778, 3)
        # lbs_faces (1552, 3)
        # exit()


        wp.sim.parse_lbs(
            builder, lbs_weights, lbs_transforms, lbs_G, lbs_rest, 
            lbs_base_transform, lbs_scale, lbs_verts, lbs_faces)


        for i in range(num_envs):
            builder.add_rigid_articulation(articulation_builder)
            if i == 0:
                self.dof_q = len(builder.joint_q)
                self.dof_qd = len(builder.joint_qd)
            coord_count = self.dof_q 
            dof_count = self.dof_qd

            print(self.dof_q, self.dof_qd)

            coord_start = i*coord_count
            dof_start = i*dof_count

            # position above ground and rotate to +y up
            builder.joint_q[coord_start:coord_start+3] = [i*2.0 + 2.3, 10.70, 1.2]
            builder.joint_q[coord_start+3:coord_start+7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), 0)
            # builder.joint_q[coord_start+3:coord_start+7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
            print("builder.joint_q", builder.joint_q)
        # finalize model
        self.model = builder.finalize(device)
        # self.model.ground = True
        self.model.joint_attach_ke *= 8.0
        self.model.joint_attach_kd *= 2.0

        
        self.integrator = wp.sim.XPBDIntegrator(
            iterations=5, joint_positional_relaxation=0.25,
            joint_angular_relaxation=0.2, angular_damping=0.1)
        
        self.integrator.contact_con_weighting = True

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/lbs_test.usd"))

 
    def run(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        if self.model.lbs_verts_count > 0:
            wp.sim.update_lbs(
                self.model,
                self.state.body_q
                )

        if (self.model.ground):
            self.model.collide(self.state)


        profiler = {}
    
        q_history = []
        q_history.append(self.state.body_q.numpy().copy())
        qd_history = []
        qd_history.append(self.state.body_qd.numpy().copy())
        delta_history = []
        delta_history.append(self.state.body_deltas.numpy().copy())
        num_con_history = []
        num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

        joint_q_history = []
        joint_q = wp.zeros_like(self.model.joint_q)
        joint_qd = wp.zeros_like(self.model.joint_qd)


        # simulate 
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            if (self.render):
 
                with wp.ScopedTimer("render", False):

                    if (self.render):
                        self.render_time += self.frame_dt
                        
                        self.renderer.begin_frame(self.render_time)
                        self.renderer.render(self.state)
                        self.renderer.end_frame()

                self.renderer.save()
                # print("render initial")
                # exit()


            print("self.sim_substeps", self.sim_substeps)
            print("self.episode_frames", self.episode_frames)
            from tqdm import trange
            for f in trange(0, self.episode_frames):
                
                for i in range(0, self.sim_substeps):
                    self.state.clear_forces()
                    
                    # keep floating base fixed
                    # wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                    # q = joint_q.numpy().copy()
                    # q[:3] = [2.3, 1.70, 1.2]
                    # q[3:7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)
                    # qd = joint_qd.numpy().copy()
                    # qd[:6] = 0.0
                    # wp.sim.eval_fk(
                    #     self.model,
                    #     wp.array(q, dtype=wp.float32, device=joint_q.device),
                    #     wp.array(qd, dtype=wp.float32, device=joint_q.device),
                    #     None,
                    #     self.state)
                    
                    random_actions = False
                    if (random_actions):
                        act = np.zeros(len(self.model.joint_qd))
                        scale = np.array([200.0,
                                        200.0,
                                        200.0,
                                        200.0,
                                        200.0,
                                        600.0,
                                        400.0,
                                        100.0,
                                        100.0,
                                        200.0,
                                        200.0,
                                        600.0,
                                        400.0,
                                        100.0,
                                        100.0,
                                        100.0,
                                        100.0,
                                        200.0,
                                        100.0,
                                        100.0,
                                        200.0])
                        for j in range(self.num_envs):
                            act[j*self.dof_qd+6:(j+1)*self.dof_qd] = np.clip((np.random.rand(self.dof_qd-6)*2.0 - 1.0)*1000.0, a_min=-1.0, a_max=1.0)*scale*0.35

                        # act[6:] = np.clip((np.random.rand(len(self.model.joint_qd)-6)*2.0 - 1.0)*1000.0, a_min=-1.0, a_max=1.0)*scale*0.35
                        # self.model.joint_act.assign(act)


                    wp.sim.collide(self.model, self.state)
                    self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                    self.sim_time += self.sim_dt


                    if self.model.lbs_verts_count > 0:
                        wp.sim.update_lbs(
                            self.model,
                            self.state.body_q
                            )

                    # rigid_contact_count = self.model.rigid_contact_count.numpy()[0]
                    # self.max_contact_count = max(self.max_contact_count, rigid_contact_count)                    
                    # if i == 0 and rigid_contact_count > 0:
                    #     qs = self.state.body_q.numpy()

                    #     contact_points_a = self.model.rigid_contact_point0.numpy()
                    #     body_a = self.model.rigid_contact_body0.numpy()
                    #     self.points_a[:rigid_contact_count] = [wp.transform_point(qs[body], wp.vec3(*contact_points_a[i])) for i, body in enumerate(body_a)]

                    #     contact_points_b = self.model.rigid_contact_point1.numpy()
                    #     body_b = self.model.rigid_contact_body1.numpy()
                    #     self.points_b[:rigid_contact_count] = [wp.transform_point(qs[body], wp.vec3(*contact_points_b[i])) for i, body in enumerate(body_b)]

                    q_history.append(self.state.body_q.numpy().copy())
                    qd_history.append(self.state.body_qd.numpy().copy())
                    delta_history.append(self.state.body_deltas.numpy().copy())
                    num_con_history.append(self.model.rigid_contact_inv_weight.numpy().copy())

                    wp.sim.eval_ik(self.model, self.state, joint_q, joint_qd)
                    joint_q_history.append(joint_q.numpy().copy())


                    if self.max_contact_count > 0:
                        rigid_contact_count = self.model.rigid_contact_count.numpy()[0]
                        self.max_contact_count = max(self.max_contact_count, rigid_contact_count)
                        
                        if rigid_contact_count > 0:
                            qs = self.state.body_q.numpy()

                            contact_points_a = self.model.rigid_contact_point0.numpy()
                            body_a = self.model.rigid_contact_body0.numpy()
                            self.points_a[:rigid_contact_count] = [(wp.transform_point(qs[body], wp.vec3(*contact_points_a[i])) if body >= 0 else wp.vec3(*contact_points_a[i])) for i, body in enumerate(body_a)]

                            contact_points_b = self.model.rigid_contact_point1.numpy()
                            body_b = self.model.rigid_contact_body1.numpy()
                            self.points_b[:rigid_contact_count] = [(wp.transform_point(qs[body], wp.vec3(*contact_points_b[i])) if body >= 0 else wp.vec3(*contact_points_b[i])) for i, body in enumerate(body_b)]


                if (self.render):
 
                    with wp.ScopedTimer("render", False):

                        if (self.render):
                            self.render_time += self.frame_dt
                            
                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state)

                            if self.max_contact_count > 0:
                                self.renderer.render_points("contact_points_a", np.array(self.points_a), radius=0.05)
                                self.renderer.render_points("contact_points_b", np.array(self.points_b), radius=0.05)

                            self.renderer.end_frame()

                    self.renderer.save()

            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        
        if False:
            import matplotlib.pyplot as plt
            q_history = np.array(q_history)
            qd_history = np.array(qd_history)
            delta_history = np.array(delta_history)
            num_con_history = np.array(num_con_history)
            # print("max num_con_history:", np.max(num_con_history))

            body_indices = [9]

            fig, ax = plt.subplots(len(body_indices), 7, figsize=(10, 10), squeeze=False)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            for i, j in enumerate(body_indices):
                ax[i,0].set_title(f"Body {j} Position")
                ax[i,0].grid()
                ax[i,1].set_title(f"Body {j} Orientation")
                ax[i,1].grid()
                ax[i,2].set_title(f"Body {j} Linear Velocity")
                ax[i,2].grid()
                ax[i,3].set_title(f"Body {j} Angular Velocity")
                ax[i,3].grid()
                ax[i,4].set_title(f"Body {j} Linear Delta")
                ax[i,4].grid()
                ax[i,5].set_title(f"Body {j} Angular Delta")
                ax[i,5].grid()
                ax[i,6].set_title(f"Body {j} Num Contacts")
                ax[i,6].grid()
                ax[i,0].plot(q_history[:,j,:3])        
                ax[i,1].plot(q_history[:,j,3:])
                ax[i,2].plot(qd_history[:,j,3:])
                ax[i,3].plot(qd_history[:,j,:3])
                ax[i,4].plot(delta_history[:,j,3:])
                ax[i,5].plot(delta_history[:,j,:3])
                ax[i,6].plot(num_con_history[:,j])
                ax[i,0].set_xlim(0, self.sim_steps)
                ax[i,1].set_xlim(0, self.sim_steps)
                ax[i,2].set_xlim(0, self.sim_steps)
                ax[i,3].set_xlim(0, self.sim_steps)
                ax[i,4].set_xlim(0, self.sim_steps)
                ax[i,5].set_xlim(0, self.sim_steps)
                ax[i,6].set_xlim(0, self.sim_steps)
                ax[i,6].yaxis.get_major_locator().set_params(integer=True)
            plt.show()

        if True:
            import matplotlib.pyplot as plt
            joint_q_history = np.array(joint_q_history)
            dof_q = joint_q_history.shape[1]
            ncols = int(np.ceil(np.sqrt(dof_q)))
            nrows = int(np.ceil(dof_q / float(ncols)))
            fig, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                constrained_layout=True,
                figsize=(ncols * 3.5, nrows * 3.5),
                squeeze=False,
                sharex=True
            )

            joint_id = 0
            joint_names = {
                wp.sim.JOINT_BALL.val : "ball", 
                wp.sim.JOINT_REVOLUTE.val : "hinge", 
                wp.sim.JOINT_PRISMATIC.val : "slide", 
                wp.sim.JOINT_UNIVERSAL.val : "universal",
                wp.sim.JOINT_COMPOUND.val : "compound",
                wp.sim.JOINT_FREE.val : "free", 
                wp.sim.JOINT_FIXED.val : "fixed"
            }
            joint_lower = self.model.joint_limit_lower.numpy()
            joint_upper = self.model.joint_limit_upper.numpy()
            joint_type = self.model.joint_type.numpy()
            while joint_id < len(joint_type)-1 and joint_type[joint_id] == wp.sim.JOINT_FIXED.val:
                # skip fixed joints
                joint_id += 1
            q_start = self.model.joint_q_start.numpy()
            qd_start = self.model.joint_qd_start.numpy()
            qd_i = qd_start[joint_id]
            for dim in range(ncols * nrows):
                ax = axes[dim // ncols, dim % ncols]
                if dim >= dof_q:
                    ax.axis("off")
                    continue
                ax.grid()
                ax.plot(joint_q_history[:, dim])
                if joint_type[joint_id] != wp.sim.JOINT_FREE.val:
                    lower = joint_lower[qd_i]
                    if abs(lower) < 2*np.pi:
                        ax.axhline(lower, color="red")
                    upper = joint_upper[qd_i]
                    if abs(upper) < 2*np.pi:
                        ax.axhline(upper, color="red")
                joint_name = joint_names[joint_type[joint_id]]
                ax.set_title(f"$\\mathbf{{q_{{{dim}}}}}$ ({self.model.joint_name[joint_id]} / {joint_name} {joint_id})")
                if joint_id < self.model.joint_count-1 and q_start[joint_id+1] == dim+1:
                    joint_id += 1
                    qd_i = qd_start[joint_id]
                else:
                    qd_i += 1
            plt.tight_layout()
            plt.show()

        return 1000.0*float(self.num_envs)/avg_time

profile = False

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):

        robot = Robot(render=False, device='cuda', num_envs=env_count)
        steps_per_second = robot.run()

        env_size.append(env_count)
        env_times.append(steps_per_second)
        
        env_count *= 2

    # dump times
    for i in range(len(env_times)):
        print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

    # plot
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(env_size, env_times)
    plt.xscale('log')
    plt.xlabel("Number of Envs")
    plt.yscale('log')
    plt.ylabel("Steps/Second") 
    plt.show()

else:

    robot = Robot(render=True, num_envs=1, device=wp.get_preferred_device())
    robot.run()
