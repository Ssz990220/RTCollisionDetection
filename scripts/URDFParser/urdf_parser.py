import xml.etree.ElementTree as ET
import numpy as np
import math
import os
import trimesh
from collections import defaultdict, deque
import argparse
import xml.dom.minidom
from scipy.spatial.transform import Rotation
import copy
from urdf_formatter import format_and_write
import json
from sphereGenUtils import *
from urdfParserUtils import *

header_base = "./RTCD/Robot/models/"
base_dir = ""
merged_mesh_dir = "" # will be set based on the passed in URDF file path, declare here for scope
sphere_dir = "" # will be set based on the passed in URDF file path, declare here for scope
link_to_meshpath = {}
link_to_sphere = {}
link_to_merged_trimesh = {}
json_links = {}

def genLinkSphere(sphere_dir):
    for link, mesh in link_to_merged_trimesh.items():
        print("Processing link: ", link, end="")
        spheres,_ = generateSphereEnvelop(mesh, maxEdgeLength=0.05, KDTreeDepth=3)
        print(" - Done")
        spheres_data = [{"center": sphere[0].tolist(), "radius": sphere[1]} for sphere in spheres]
        json_links[link]["spheres"] = spheres_data
        # save the spheres data to a file
        with open(os.path.join(sphere_dir, link + ".txt"), "w") as f:
            for sphere in spheres_data:
                f.write(f"{sphere['center'][0]} {sphere['center'][1]} {sphere['center'][2]} {sphere['radius']}\n")
        print("Generated spheres for link: ", link)
        print("Spheres: ", spheres_data)

# A robot kinematic chain and its geometry can be represented as a JSON file.
# The link is in breath-first order based on the tree structure of mechanism.
# So does the joint.
# Each joint is represented by three 3x4 matrices: sinMask, cosMask, oneMask.
# Each joint has its type: 0 for revolute, 1 for prismatic.
# Each joint has its parent link index.
# Joint i read the mask from [i*12, i*12+11], the transformation is applied to the preTfIdx-th link tf
# The baseT is the base transformation of the robot, which can be None that indicates a floating base.
def generate_json_file(filepath, robotName, nLinks,
                    dof, baseT, lowerBound, upperBound,
                    all_sinMask, all_cosMask, all_oneMask,
                    jointTypes, jointPreTfIdx):
    
    # json data:
    data = {
        "robotName": robotName,
        "nLinks": nLinks,
        "dof": dof,
        "baseT": baseT,
        "lowerBound": lowerBound,
        "upperBound": upperBound,
        "jointTypes": jointTypes,
        "jointPreTfIdx": jointPreTfIdx,
        "links": json_links,
        "sinMask": all_sinMask,
        "cosMask": all_cosMask,
        "oneMask": all_oneMask
    }

    # write to file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    

############################################
if __name__ == '__main__':

    ## As a first step, we locate the URDF file and parse it
    parser = argparse.ArgumentParser(description="Process a URDF file to merge fixed joints and generate header file.")
    parser.add_argument('--urdf_file', type=str, default='./models/franka_description/panda.urdf', help='Path to the URDF file')
    parser.add_argument("--gen_sphere", action='store_true', help="Generate sphere files")
    args = parser.parse_args()

    urdf_file = args.urdf_file
    robotName = os.path.splitext(os.path.basename(urdf_file))[0]
    base_dir = os.path.dirname(urdf_file)
    merged_mesh_dir = os.path.join(base_dir, "merged_meshes")
    sphere_dir = os.path.join(base_dir, "spheres")

    if not os.path.exists(merged_mesh_dir):
        os.makedirs(merged_mesh_dir)

    if not os.path.exists(sphere_dir):
        os.makedirs(sphere_dir)

    # Parse URDF
    root, tree = parse_urdf(urdf_file)

    # Create a new URDF tree but with only the robot tag
    new_root = ET.Element('robot', attrib=root.attrib)

    # Create an empty dictionary to store link_name to link_mesh map
    link_mesh_map = {}
    
    for link in root.findall('link'):
        link_mesh_map[link.get('name')] = process_link_geometry(link, os.path.dirname(os.path.abspath(urdf_file)))

    # Remove fixed joints and merge links
    links, joints, graph, root_link = build_graph(root)
    fixed_groups = find_fixed_groups(links, graph)
    link_ref_tf = merge_fixed_links(fixed_groups, links, graph, link_mesh_map, new_root, base_dir, link_to_meshpath, link_to_merged_trimesh, merged_mesh_dir)

    rewire_joints(links, joints, graph, link_ref_tf, new_root, root_link)

    modified_urdf_file = os.path.join(base_dir, "modified_" + os.path.basename(urdf_file))

    # write the new urdf to a file with format to enhance readability
    dom = xml.dom.minidom.parseString(ET.tostring(new_root))
    pretty_xml_as_string = dom.toprettyxml()

    with open(modified_urdf_file, "w") as f:
        f.write(pretty_xml_as_string)
    

    ## Count the number of DOF in the modified URDF
    mod_root, mod_tree = parse_urdf(modified_urdf_file)
    dof = count_dof(mod_root)
    nLinks = len(mod_root.findall('link'))
    mod_links, mod_joints, mod_graph, mod_root_link = build_graph(mod_root)
    reorder_elements(mod_root_link, mod_graph, mod_root, json_links, link_to_meshpath)
    ## rewrite with reordering
    dom = xml.dom.minidom.parseString(ET.tostring(mod_root))
    pretty_xml_as_string = dom.toprettyxml()

    with open(modified_urdf_file, "w") as f:
        f.write(pretty_xml_as_string)

    format_and_write(modified_urdf_file)

    ## Generate joint & link masks
    linkIdx = {}
    idx = 0
    for link in mod_root.findall('link'):
        linkIdx[link.get('name')] = idx
        idx += 1

    if args.gen_sphere:
        genLinkSphere(sphere_dir)
    
    all_cosMask = []
    all_sinMask = []
    all_oneMask = []
    lowerBound = []
    upperBound = []
    jointTypes = []
    jointPreTfIdx = []
    for joint in mod_root.findall('joint'):
        cM, sM, oM = generate_masks_for_joint(joint)
        all_cosMask.extend(cM)
        all_sinMask.extend(sM)
        all_oneMask.extend(oM)
        jointTypes.append(jointType(joint))
        jointPreTfIdx.append(linkIdx[joint.find('parent').get('link')])

        limit = joint.find('limit')
        if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
            lower = float(limit.attrib['lower'])
            upper = float(limit.attrib['upper'])
            lowerBound.append(lower)
            upperBound.append(upper)
        else:
            # If no limit is specified, set some default or raise an error
            # For now, assume all non-fixed joints have limits.
            raise ValueError(f"No limit specified for joint {joint.get('name')}")

    baseT = [1.0,0.0,0.0,0.0,
             0.0,1.0,0.0,0.0,
             0.0,0.0,1.0,0.0]

    generate_header_file(header_base, nLinks, urdf_file,
                         dof, baseT, lowerBound, upperBound,
                         all_sinMask, all_cosMask, all_oneMask,
                         jointTypes, jointPreTfIdx)
    # generate_json_file(f"{base_dir}/{robotName.lower()}.json", robotName, nLinks,
    #                 dof, baseT, lowerBound, upperBound,
    #                 all_sinMask, all_cosMask, all_oneMask,
    #                 jointTypes, jointPreTfIdx)
    
    print("cosMask = {")
    print(", ".join(map(str, all_cosMask)))
    print("};\n")

    print("sinMask = {")
    print(", ".join(map(str, all_sinMask)))
    print("};\n")

    print("oneMask = {")
    print(", ".join(map(str, all_oneMask)))
    print("};\n")

    dof_count = count_dof(mod_root)
    print(f"Degrees of Freedom: {dof_count}")
    
    print("Lower Bound = {")
    print(", ".join(map(str, lowerBound)))
    print("};\n")

    print("Upper Bound = {")
    print(", ".join(map(str, upperBound)))
    print("};\n")

    print("Joint Types = {")
    print(", ".join(map(str, jointTypes)))
    print("};\n")

    print("Joint Pre-Tf Idx = {")
    print(", ".join(map(str, jointPreTfIdx)))
    print("};\n")