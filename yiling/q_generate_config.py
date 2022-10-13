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
from q_mano import QManoLayer

import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def main():
    mano_path='externals/mano/'
    mano_layer_right = QManoLayer(
        mano_root=mano_path, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

    pose, trans = torch.zeros([1, 48]), torch.zeros([1, 3])

    verts, joints = mano_layer_right(pose, th_trans=trans) # in MM
    faces = mano_layer_right.th_faces

    print(joints.shape)

    joints = joints * 20 + torch.tensor([[[0, 0, 200]]])

    # mesh_hand = mesh_utils.Meshes(verts, faces.unsqueeze(0))
    # mesh_utils.dump_meshes(osp.join("output", 'test_hand_zero'), mesh_hand)

    # parse
    filename = 'hand_1finger.xml'
    file = ET.parse(filename)
    root = file.getroot()

    world = root.find("worldbody")
    # hand_links = [
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8],
    #     [9, 10, 11, 12],
    #     [13, 14, 15, 16],
    #     [17, 18, 19, 20],
    #     ]
    hand_links = [
        [1, 2, 3, 17],
        [4, 5, 6, 18],
        [7, 8, 9, 20],
        [10, 11, 12, 19],
        [13, 14, 15, 16],
        ]
    # [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    # hand_links = [
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    #     [13, 14, 15],
    #     [10, 11, 12],
    #     ]
    # links = [0, 7, 8, 9, 20]
    # 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20

    for body in world.findall("body"):
        world.remove(body)
       
    body = ET.SubElement(world, 'body')
    attrib = {
        'name': 'palm', 
        # 'pos': '{:.04f} {:.04f} {:.04f}'.format(joints[0,0,0], joints[0,0,1], joints[0,0,2]+5),
        'pos': '0 0 5',
        # 'quat': '0 0 0 1',
        'quat': '1 0 0 0',
        }
    body.attrib = attrib

    joint = ET.SubElement(body, 'joint')
    attrib = {
        'name': 'root', 
        'armature': '1',
        'damping': '10',
        'limited': 'false',
        'margin': '0.01',
        'pos': '0 0 0',
        'type': 'free',
        }
    joint.attrib = attrib
    geom = ET.SubElement(body, 'geom')
    geom_attrib = {
        'name': 'palm', 
        'type': 'sphere',
        'size': '0.1',
        'pos': '0 0 0',
    }
    geom.attrib = geom_attrib
    # geom = ET.SubElement(body, 'geom')
    # geom_attrib = {
    #     'name': 'palm', 
    #     'type': 'capsule',
    #     'size': '0.1',
    #     'fromto': '0 0 0 1 0 0',
    # }
    # geom.attrib = geom_attrib

    for i_finger  in range(len(hand_links)):
        finger = hand_links[i_finger]
        parent = body
        
        last_link = 0
        for i_link in range(len(finger) - 1):
            this_link = finger[i_link]
            next_link = finger[i_link+1]
            name = 'f{}j{}'.format(i_finger, this_link)
            
            father_offset = joints[0,this_link] - joints[0,last_link]
            
            body_i = ET.SubElement(parent, 'body')
            attrib = {
                'name': name, 
                # 'quat': '0 0 0 1',
                'quat': '1 0 0 0',
                'pos': '{:.04f} {:.04f} {:.04f}'.format(father_offset[0], father_offset[1], father_offset[2]),
            }
            body_i.attrib = attrib
            
            geom = ET.SubElement(body_i, 'geom')
            offset = joints[0,next_link] - joints[0,this_link]
            geom_offset = offset * 1.0
            attrib = {
                'name': name, 
                'type': 'capsule',
                'size': '0.1',
                # 'fromto': '0 0 0 0.1 0 0',
                'fromto': ' 0 0 0 {:.04f} {:.04f} {:.04f}'.format(geom_offset[0], geom_offset[1], geom_offset[2]),
            }
            geom.attrib = attrib
            # geom = ET.SubElement(body_i, 'geom')
            # attrib = {
            #     'name': name, 
            #     'type': 'sphere',
            #     'size': '0.1',
            #     'pos': '0 0 0',
            # }
            # geom.attrib = attrib

            # joint = ET.SubElement(body_i, 'joint')
            # attrib = {
            #     'name': name + "_x", 
            #     'type': 'ball',
            #     'armature': '1.0',
            #     'damping': '10',
            #     'pos': '0 0 0',
            #     'range': '0 0'
            #     # 'pos': '{:.04f} {:.04f} {:.04f}'.format(-offset[0], -offset[1], -offset[2])
            #     }
            # joint.attrib = attrib

            joint = ET.SubElement(body_i, 'joint')
            attrib = {
                'name': name + "_x", 
                'armature': '1.0',
                'damping': '10',
                'limited': 'false',
                'axis': '1 0 0',
                'range': '-90 90',
                'margin': '0.01',
                'pos': '0 0 0'
                # 'pos': '{:.04f} {:.04f} {:.04f}'.format(-offset[0], -offset[1], -offset[2])
                }
            joint.attrib = attrib

            # joint = ET.SubElement(body_i, 'joint')
            # attrib = {
            #     'name': name + "_y", 
            #     'armature': '1.0',
            #     'damping': '10',
            #     'limited': 'false',
            #     'axis': '0 1 0',
            #     'range': '-90 90',
            #     'margin': '0.01',
            #     'pos': '0 0 0'
            #     }
            # joint.attrib = attrib

            # joint = ET.SubElement(body_i, 'joint')
            # attrib = {
            #     'name': name + "_z", 
            #     'armature': '1.0',
            #     'damping': '10',
            #     'limited': 'false',
            #     'axis': '0 0 1',
            #     'range': '-90 90',
            #     'margin': '0.01',
            #     'pos': '0 0 0'
            #     }
            # joint.attrib = attrib

            parent = body_i
            last_link = this_link


    # for i_link in links:
    #     name = 'finger_joint_{}'.format(i_link)
        
    #     geom = ET.SubElement(body, 'geom')
    #     geom_attrib = {
    #         'name': name, 
    #         'type': 'sphere',
    #         'size': '0.1',
    #         'pos': '{:.04f} {:.04f} {:.04f}'.format(joints[0,i_link,0], joints[0,i_link,1], joints[0,i_link,2]),
    #     }
    #     geom.attrib = geom_attrib
    #     # body = body_next

    tree = ET.ElementTree(root)
    
    indent(root)

    with open('test2.xml', 'w') as f:
        tree.write(f, encoding='unicode')





if __name__ == '__main__':
    main()





