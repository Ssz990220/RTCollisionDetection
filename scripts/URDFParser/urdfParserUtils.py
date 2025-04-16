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
import json

############################################
# Utility and Parsing Functions
############################################
class linkt:
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []
        self.ref = None
        self.leaves = set()

    def set_ref(self, ref):
        self.ref = ref

    def set_leaves(self, leaves):
        self.leaves = leaves

def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root, tree

def get_origin_transform(elem):
    rpy = [0,0,0]
    xyz = [0,0,0]
    origin = elem.find('origin')
    if origin is not None:
        if 'rpy' in origin.attrib:
            rpy = list(map(float, origin.attrib['rpy'].split()))
        if 'xyz' in origin.attrib:
            xyz = list(map(float, origin.attrib['xyz'].split()))
    R = Rotation.from_euler('xyz', rpy)
    return R, xyz



def get_origin_transform_matrix(elem):
    rpy = [0,0,0]
    xyz = [0,0,0]
    origin = elem.find('origin')
    if origin is not None:
        if 'rpy' in origin.attrib:
            rpy = list(map(float, origin.attrib['rpy'].split()))
        if 'xyz' in origin.attrib:
            xyz = list(map(float, origin.attrib['xyz'].split()))
    R = Rotation.from_euler('xyz', rpy)
    T = np.eye(4)
    T[0:3, 0:3] = R.as_matrix()
    T[0:3, 3] = xyz
    return T

def get_scale(mesh_elem):
    # scale attribute can appear in <mesh scale="x y z">
    # If not present, scale is (1,1,1)
    if 'scale' in mesh_elem.attrib:
        s = list(map(float, mesh_elem.attrib['scale'].split()))
        return np.array(s)
    return np.array([1.0, 1.0, 1.0])

def load_mesh_with_transform(base_dir, mesh_path_rel, origin_R, origin_v, scale_vector):
    # Resolve path
    origin_transform = np.eye(4)
    origin_transform[0:3, 0:3] = origin_R.as_matrix()
    origin_transform[0:3, 3] = origin_v
    mesh_path_abs = mesh_path_rel
    if not os.path.isabs(mesh_path_abs):
        mesh_path_abs = os.path.join(base_dir, mesh_path_abs)
    # Load mesh
    mesh = trimesh.load(mesh_path_abs, force="mesh", skip_materials=True)
    mesh.merge_vertices(merge_norm = True)
    # Apply scale
    mesh.apply_scale(scale_vector)
    # Apply origin transform
    mesh.apply_transform(origin_transform)
    return mesh, mesh_path_rel

############################################
# Merge Links in a Rigid Group
############################################

def process_link_geometry(link_elem, base_dir):
    """
    Given an XML <link> element, parse its <visual> and <collision> geometry,
    load or generate trimesh meshes, apply any origin and scale transforms,
    and merge everything into a single mesh.

    Args:
      link_elem: XML element for <link>
      base_dir: Base directory for resolving mesh paths

    Returns:
        Tuple of (merged_mesh_viz, merged_mesh_col) where each is a trimesh object
    """
    merged_mesh_viz = None
    merged_mesh_col = None

    # --------------------------------------------------------------------------
    # Helper function to merge a new mesh into the cumulative 'merged_mesh_*'
    # --------------------------------------------------------------------------
    def merge_in(new_mesh, existing_mesh):
        if new_mesh is None:
            return existing_mesh
        if existing_mesh is None:
            return new_mesh
        return trimesh.util.concatenate([existing_mesh, new_mesh])

    # --------------------------------------------------------------------------
    # Helper function to create a primitive shape from a URDF <geometry> element
    # (box, cylinder, sphere) and apply a transform.
    # --------------------------------------------------------------------------
    def create_primitive_mesh(geom_elem, origin_R, origin_v):
        box_elem = geom_elem.find('box')
        origin_transform = np.eye(4)
        origin_transform[0:3, 0:3] = origin_R.as_matrix()
        origin_transform[0:3, 3] = origin_v
        if box_elem is not None:
            size_str = box_elem.attrib.get('size', '')
            if size_str:
                size = list(map(float, size_str.split()))
                prim_mesh = trimesh.creation.box(extents=size)
                prim_mesh.apply_transform(origin_transform)
                return prim_mesh
            return None

        cyl_elem = geom_elem.find('cylinder')
        if cyl_elem is not None:
            radius = float(cyl_elem.attrib.get('radius', '0'))
            length = float(cyl_elem.attrib.get('length', '0'))
            prim_mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=32)
            prim_mesh.apply_transform(origin_transform)
            return prim_mesh

        sphere_elem = geom_elem.find('sphere')
        if sphere_elem is not None:
            radius = float(sphere_elem.attrib.get('radius', '0'))
            prim_mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            prim_mesh.apply_transform(origin_transform)
            return prim_mesh

        # If no recognized primitive, return None
        return None

    # --------------------------------------------------------------------------
    # Process <visual> elements
    # --------------------------------------------------------------------------
    for visual_elem in link_elem.findall('visual'):
        geom_elem = visual_elem.find('geometry')
        if geom_elem is None:
            continue

        # Get the 4x4 origin transform from <origin xyz="..." rpy="...">
        origin_R, origin_v = get_origin_transform(visual_elem)

        # Check if this geometry is a <mesh> or a primitive (<box>, <cylinder>, <sphere>, etc.)
        mesh_elem = geom_elem.find('mesh')
        if mesh_elem is not None:
            # If it's a mesh, use the provided helper to load and transform it
            scale_vector = get_scale(mesh_elem)
            mesh_path_rel = mesh_elem.attrib.get('filename', '')
            if mesh_path_rel:
                new_mesh, _ = load_mesh_with_transform(base_dir, mesh_path_rel, origin_R, origin_v, scale_vector)
                merged_mesh_viz = merge_in(new_mesh, merged_mesh_viz)
        else:
            # Otherwise, see if it's a primitive shape
            prim_mesh = create_primitive_mesh(geom_elem, origin_R, origin_v)
            merged_mesh_viz = merge_in(prim_mesh, merged_mesh_viz)

    # --------------------------------------------------------------------------
    # Process <collision> elements
    # (same logic as above, or you can choose to handle them separately)
    # --------------------------------------------------------------------------
    for collision_elem in link_elem.findall('collision'):
        geom_elem = collision_elem.find('geometry')
        if geom_elem is None:
            continue

        origin_R, origin_v = get_origin_transform(collision_elem)
        
        mesh_elem = geom_elem.find('mesh')
        if mesh_elem is not None:
            scale_vector = get_scale(mesh_elem)
            mesh_path_rel = mesh_elem.attrib.get('filename', '')
            if mesh_path_rel:
                new_mesh, _ = load_mesh_with_transform(base_dir, mesh_path_rel, origin_R, origin_v, scale_vector)
                merged_mesh_col = merge_in(new_mesh, merged_mesh_col)
        else:
            prim_mesh = create_primitive_mesh(geom_elem, origin_R, origin_v)
            merged_mesh_col = merge_in(prim_mesh, merged_mesh_col)

    return (merged_mesh_viz, merged_mesh_col)

# merge_group_into_single_link(group_links, links, graph, root, base_dir)
# merge a group of links into a single link (if the group contains more than one link)
# groups: a set of link names to merge
# links: a map between the link name and the link element in the tree, we use this to retrieve the link information
# graph: a map between the link name and its neighbors, we use this to traverse the tree
# new_root: the root of the new URDF, we append this tree with the new link
# link_mesh_map: a dictionary mapping link names to its pre-procesed mesh
# 
# Returns: A link transformation map that tracks the transformation caused by fixed joints
#
# Logic:
# Iterate over all group in groups
# For each group, we need to:
# merge meshes binded to each link in the group
#  - inertial properties are merged if provided
#  - visual and collision meshes are merged
#  - transformation caused by fixed joints are tracked for joint rewiring
#  - the merged link inherits the name of the first link in the group
def merge_fixed_links(groups, links, graph, link_mesh_map, new_root, base_dir, link_to_meshpath, link_to_merged_trimesh, merged_mesh_dir):
    link_tf_to_ref = {}
    for group in groups:
        # reorder the group so that the first link is the closest to the root
        # we will use this link as the reference link
        # the first link is the one that has parent link outside the group
        if len(group) == 1: # A single link, the new link will completely inherit the old link
            for g in group:
                link_tf_to_ref[g] = np.eye(4)

                graph[g].set_ref(g)
                graph[g].set_leaves({g})
                ref_link = g
        else: # Multiple links
            # we need to find the reference link that is the closest to the root link
            # we find this by checking if the parent link is outside the group
            ref_link = None
            leaf_links = set()

            for link in group:
                if graph[link].parent is None: # the link is the root
                    ref_link = link
                else:
                    if graph[link].parent[0] not in group: # covers the case that the link is the parent
                        ref_link = link
                
                if len(graph[link].children) == 0:
                    leaf_links.add(link)
                else:
                    for child, _ in graph[link].children:
                        if child not in group:
                            leaf_links.add(link)
                            break
            
            for link in group:
                graph[link].set_ref(ref_link)
            
            graph[ref_link].set_leaves(leaf_links)

            # track the transformation of the reference link
            link_tf_to_ref[ref_link] = np.eye(4)

            def compute_link_tf_to_ref(link, link_tf_to_ref):
                if link in link_tf_to_ref:
                    return link_tf_to_ref[link]
                parent_link, joint = graph[link].parent
                link_tf_to_ref[link] = compute_link_tf_to_ref(parent_link, link_tf_to_ref) @ get_origin_transform_matrix(joint)
                return link_tf_to_ref[link]

            
            for link in group:
                compute_link_tf_to_ref(link, link_tf_to_ref)

        # merge meshes
        merged_visual = None
        merged_collision = None

        # save the merged meshes
        merged_mesh_path = os.path.join(merged_mesh_dir, ref_link).replace("\\", "/")
        for link in group:
            meshViz, meshCol = link_mesh_map[link]
            if meshViz is not None:
                if merged_visual is None:
                    merged_visual = meshViz.apply_transform(link_tf_to_ref[link])
                else:
                    merged_visual = trimesh.util.concatenate([merged_visual, meshViz.apply_transform(link_tf_to_ref[link])])

            if meshCol is not None:
                if merged_collision is None:
                    merged_collision = meshCol.apply_transform(link_tf_to_ref[link])
                else:
                    merged_collision = trimesh.util.concatenate([merged_collision, meshCol.apply_transform(link_tf_to_ref[link])])
        

        # start with the ref link
        new_link = ET.Element('link', attrib={'name': ref_link})
        
        if merged_visual:
            merged_visual.export(merged_mesh_path + "_visual.obj")

            visual = ET.Element('visual')
            visual_origin = ET.Element('origin', attrib={'xyz': '0 0 0', 'rpy': '0 0 0'})
            visual_geom = ET.Element('geometry')
            mesh_path = os.path.relpath(merged_mesh_path, base_dir).replace("\\", "/")
            visual_mesh = ET.Element('mesh', attrib={'filename': mesh_path + "_visual.obj"})
            visual_geom.append(visual_mesh)
            visual.append(visual_origin)
            visual.append(visual_geom)

            new_link.append(visual)
            
            link_to_meshpath[ref_link] = os.path.basename(mesh_path + "_visual.obj")
            link_to_merged_trimesh[ref_link] = merged_visual
        else:
            # raise warning
            print(f"Warning: No visual mesh found for link {ref_link}")

        if merged_collision:
            merged_collision.export(merged_mesh_path + "_collision.obj")
            
            collision = ET.Element('collision')
            collision_origin = ET.Element('origin', attrib={'xyz': '0 0 0', 'rpy': '0 0 0'})
            collision_geom = ET.Element('geometry')
            mesh_path = os.path.relpath(merged_mesh_path, base_dir).replace("\\", "/")
            collision_mesh = ET.Element('mesh', attrib={'filename': mesh_path + "_collision.obj"})
            collision_geom.append(collision_mesh)
            collision.append(collision_origin)
            collision.append(collision_geom)

            new_link.append(collision)

        new_root.append(new_link)
        print_element(new_link)
        ## TODO: merge inertial properties
    return link_tf_to_ref



def build_graph(root):
    links = {}
    for link in root.findall('link'):
        links[link.get('name')] = link

    graph = defaultdict(list)
    joints = root.findall('joint')

    graph = {}
    
    for link_elem in root.findall('link'):
        link_name = link_elem.get('name')
        # Create and store the linkt object
        graph[link_name] = linkt(link_name)
    
    for joint_elem in root.findall('joint'):
        # The <parent> and <child> tags specify which links this joint connects
        parent_name = joint_elem.find('parent').get('link')
        child_name  = joint_elem.find('child').get('link')
        
        # Retrieve the linkt objects for parent and child
        parent_link_obj = graph[parent_name]
        child_link_obj  = graph[child_name]
        
        # Update their relationship
        child_link_obj.parent = (parent_name, joint_elem)
        parent_link_obj.children.append((child_name, joint_elem))

    for link in graph.keys():
        if graph[link].parent is None:
            root = link
            break

    return links, joints, graph, root


def jointType(joint):
    T = joint.get('type')
    if T == 'revolute':
        return 0
    if T == 'prismatic':
        return 1
    else:
        return -1

def find_fixed_groups(links, graph):
    visited = set()
    groups = []
    def is_fixed(j):
        return j.get('type') == 'fixed'

    for ln in links.keys():
        if ln not in visited:
            queue = deque([ln])
            group = set()
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                group.add(curr)
                if graph[curr].parent is not None:
                    if is_fixed(graph[curr].parent[1]):
                        queue.append(graph[curr].parent[0])
                for nbr, joint in graph[curr].children:
                    if is_fixed(joint):
                        queue.append(nbr)
            groups.append(group)
    return groups

def rewire_joints(links, joints, graph, link_tf_to_ref, new_root, root_link):

    open_link = deque([root_link])

    while open_link:
        curr = open_link.popleft()
        ln = graph[curr]

        for leaf in ln.leaves:
            if leaf == curr:
                continue
            open_link.append(leaf)

        for child, joint in ln.children:
            if joint.get('type') == 'fixed':
                continue
            open_link.append(child)
            new_joint = ET.Element('joint', attrib=joint.attrib)

            ref = ln.ref
            link_tf = link_tf_to_ref[curr]

            joint_tf = get_origin_transform_matrix(joint)
            joint_tf = link_tf @ joint_tf

            origin_elem = ET.Element('origin')
            
            rpy = Rotation.from_matrix(joint_tf[0:3, 0:3]).as_euler('xyz')
            origin_elem.attrib['rpy'] = ' '.join(map(str, rpy))
            origin_elem.attrib['xyz'] = ' '.join(map(str, joint_tf[0:3,3]))
            parent_elem = ET.Element('parent', attrib={'link': ref})
            child_elem = ET.Element('child', attrib={'link': child})

            # inherit all other attributes
            for key, value in joint.attrib.items():
                if key not in ['name', 'type']:
                    new_joint.attrib[key] = value

            new_joint.append(parent_elem)
            new_joint.append(child_elem)
            new_joint.append(origin_elem)

            # inherit all other elements
            for elem in joint:
                if elem.tag not in ['origin', 'parent', 'child']:
                    new_element = ET.Element(elem.tag)
                    for key, value in elem.attrib.items():
                        new_element.attrib[key] = value
                    new_joint.append(new_element)
            
            new_root.append(new_joint)
            print_element(new_joint)    

def reorder_elements(root_link, graph, new_root, json_links, link_to_meshpath):
    # reorder the joints and links so that the order of links and joints reflects breadth first traversal
    # we will use this order to generate the masks, joint ids, and linkTfIdx
    ordered_elements = []
    open_link = deque([root_link])

    lk_name_to_elem = {}
    lks = new_root.findall('link')
    for lk in lks:
        lk_name_to_elem[lk.get('name')] = lk

    jt_name_to_elem = {}
    jts = new_root.findall('joint')
    for jt in jts:
        jt_name_to_elem[jt.get('name')] = jt

    while open_link:
        curr = open_link.popleft()
        ln = graph[curr]
        lk = lk_name_to_elem[curr]
        ordered_elements.append(copy.deepcopy(lk))
        json_links[curr] = {
                "name": curr,
                "mesh": link_to_meshpath[curr]
            }

        for child, joint in ln.children:
            open_link.append(child)
            ordered_elements.append(copy.deepcopy(jt_name_to_elem[joint.get('name')]))

    # remove all previous joints and links
    for lk in lks:
        new_root.remove(lk)
    for jt in jts:
        new_root.remove(jt)

    # add the ordered elements
    for elem in ordered_elements:
        new_root.append(elem)
            
def print_element(element, level=0):
    indent = "  " * level
    print(f"{indent}<{element.tag}", end="")
    if element.attrib:
        for key, value in element.attrib.items():
            print(f' {key}="{value}"', end="")
    print(">")
    if element.text and element.text.strip():
        print(f"{indent}  {element.text.strip()}")
    for child in element:
        print_element(child, level + 1)
    print(f"{indent}</{element.tag}>")

############################################
# Generating cos/sin/one masks
############################################

def skew(u):
    return np.array([
        [0,    -u[2], u[1]],
        [u[2],    0, -u[0]],
        [-u[1], u[0], 0]
    ])

def generate_masks_for_joint(joint):
    jtype = joint.get('type')
    if jtype not in ['revolute', 'prismatic']:
        return [], [], []

    origin = get_origin_transform_matrix(joint)
    axis_elem = joint.find('axis')
    if axis_elem is not None:
        axis = list(map(float, axis_elem.attrib['xyz'].split()))
    else:
        axis = [0,0,1]

    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    K = skew(axis)
    K2 = K @ K
    I = np.eye(3)

    # R(theta) = I + sin(theta)*K + (1 - cos(theta))*K2
    one_mat = I + K2
    sin_mat = K
    cos_mat = -K2

    R_const = origin[0:3,0:3]
    xyz = origin[0:3,3]

    T_const = np.eye(4)
    T_const[0:3,0:3] = R_const
    T_const[0:3,3] = xyz

    oneMask, cosMask, sinMask = mat_compose(T_const, one_mat, cos_mat, sin_mat)

    out_oneMask = []
    out_cosMask = []
    out_sinMask = []
    for i in range(3):
        for j in range(4):
            # if any number's abs value is less than 1e-6, we set it to 0

            def round_up(val, threshold=1e-5):
                # if close to 0, set to 0
                # if close to 1 set to 1
                # if close to -1 set to -1
                if abs(val) < threshold:
                    return 0.0
                if abs(val - 1) < threshold:
                    return 1.0
                if abs(val + 1) < threshold:
                    return -1.0
                return val
            
            out_oneMask.append(round_up(oneMask[i,j]))
            out_cosMask.append(round_up(cosMask[i,j]))
            out_sinMask.append(round_up(sinMask[i,j]))

    return out_cosMask, out_sinMask, out_oneMask

def mat_compose(T_const, one_mat, cos_mat, sin_mat):
    oneMask = np.zeros((4,4))
    cosMask = np.zeros((4,4))
    sinMask = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            if j < 3 and i < 3:
                val_one = 0.0
                val_cos = 0.0
                val_sin = 0.0
                for k in range(3):
                    val_one += T_const[i,k]*one_mat[k,j]
                    val_cos += T_const[i,k]*cos_mat[k,j]
                    val_sin += T_const[i,k]*sin_mat[k,j]
                oneMask[i,j] = val_one
                cosMask[i,j] = val_cos
                sinMask[i,j] = val_sin
            elif j == 3 and i < 3:
                oneMask[i,j] = T_const[i,3]
            elif i == 3 and j == 3:
                oneMask[i,j] = 1.0
    return oneMask, cosMask, sinMask


def count_dof(root):
    dof_count = 0
    for j in root.findall('joint'):
        jtype = j.get('type')
        if jtype in ['revolute', 'prismatic']:
            dof_count += 1
    return dof_count

def generate_header_file(header_base, nLinks, urdf_path, Dim, baseT, lowerBound, upperBound,
                         sinMask, cosMask, oneMask, jointTypes, jointPreTfIdx):
    """
    Generate a .h file in the specified format.

    Parameters:
        urdf_path (str): Path to the robot's URDF file.
        Dim (int): Number of joints.
        baseT (list or array of float): 12 floats for base transform.
        lowerBound (list of float): length Dim
        upperBound (list of float): length Dim
        sinMask (list of float): length Dim*12
        cosMask (list of float): length Dim*12
        oneMask (list of float): length Dim*12
    """
    # Helper to format arrays nicely
    def format_array(arr, elements_per_line=12):
        lines = []
        for i in range(0, len(arr), elements_per_line):
            chunk = arr[i:i+elements_per_line]
            line = ", ".join(f"{x}f" for x in chunk)
            lines.append(line)
        return ",\n    ".join(lines)


    # extract the name of the urdf from the urdf_path
    urdf_name = os.path.basename(urdf_path)
    # remove the .urdf extension
    robot_name = os.path.splitext(urdf_name)[0]
    # extract the basepath from the path of the urdf
    base_path = os.path.dirname(urdf_path)
    # append mesh and sphere base paths
    mesh_base_path = base_path + "/merged_meshes"
    sphere_base_path = base_path + "/spheres"
    header_file_path = f"{header_base}/{robot_name.lower()}.h"
    # remove the dot in the path
    urdf_base_path = base_path + "/modified_" + urdf_name
    urdf_base_path = urdf_base_path.replace("./", "/")
    mesh_base_path = mesh_base_path.replace("./", "/")
    sphere_base_path = sphere_base_path.replace("./", "/")
    

    # Convert paths to macro form:
    # We assume PROJECT_BASE_DIR is defined at compile time.
    # The user provided macros:
    # CONCAT_PATHS(PROJECT_BASE_DIR, "/models/franka_description/panda.urdf"),
    # etc.
    # Adjust urdf_path, mesh_base_path, sphere_base_path to be relative to PROJECT_BASE_DIR as needed.
    # For simplicity, assume they are relative paths from PROJECT_BASE_DIR.
    urdf_macro = f'CONCAT_PATHS(PROJECT_BASE_DIR, "{urdf_base_path}")'
    mesh_macro = f'CONCAT_PATHS(PROJECT_BASE_DIR, "{mesh_base_path}")'
    sphere_macro = f'CONCAT_PATHS(PROJECT_BASE_DIR, "{sphere_base_path}")'

    # Format arrays
    baseT_str = format_array(baseT, 12)
    sinMask_str = format_array(sinMask, 12)
    cosMask_str = format_array(cosMask, 12)
    oneMask_str = format_array(oneMask, 12)

    # Create header content
    header_content = f"""
#pragma once
#include <array>
#include <robot/robotConfig.h>
#include "config.h"

inline constexpr uint8_t Dim = {Dim};
inline constexpr uint8_t nLinks = {nLinks};
inline constexpr std::array<float, 12> baseT{{ {baseT_str} }};

inline constexpr auto lowerBound = std::array<float, Dim>{{ {", ".join(f"{x}f" for x in lowerBound)} }};
inline constexpr auto upperBound = std::array<float, Dim>{{ {", ".join(f"{x}f" for x in upperBound)} }};
inline constexpr auto jointTypes = std::array<uint8_t, Dim>{{ {", ".join(f"{x}" for x in jointTypes)} }};
inline constexpr auto jointPreTfIdx = std::array<uint8_t, Dim>{{ {", ".join(f"{x}" for x in jointPreTfIdx)} }};

inline constexpr std::array<float,Dim*12> sinMask{{
    {sinMask_str}
}};

inline constexpr std::array<float,Dim*12> cosMask{{
    {cosMask_str}
}};

inline constexpr std::array<float,Dim*12> oneMask{{
    {oneMask_str}
}};

inline constexpr RTCD::robotConfig<Dim> Config{{
    "{robot_name}",
    {urdf_macro},
    {mesh_macro},
    {sphere_macro},
    baseT,
    sinMask,
    cosMask,
    oneMask
}};
"""
    # Write to file
    with open(header_file_path, 'w') as f:
        f.write(header_content)
    print(f"Header file generated at {header_file_path}")

