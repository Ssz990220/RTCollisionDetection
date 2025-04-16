## This script load a urdf file in XML format and format it to be more readable

import xml.etree.ElementTree as ET
import argparse
import xml.dom.minidom

def format_urdf(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Format the XML file by loading element recursively in the tree and writing it to a new tree as a ET.Element object
    def format_element(element, indent=0):
        # Create a new element with the same tag
        formatted_element = ET.Element(element.tag)
        # Copy the attributes
        formatted_element.attrib = element.attrib
        # Recursively format the children
        for child in element:
            formatted_child = format_element(child, indent + 1)
            formatted_element.append(formatted_child)
        return formatted_element
    
    formatted_urdf = format_element(root)
    return formatted_urdf

def format_and_write(urdf_file):
    formatted_urdf = format_urdf(urdf_file)
    tree = ET.ElementTree(formatted_urdf)
    dom = xml.dom.minidom.parseString(ET.tostring(tree.getroot()))
    pretty_xml_as_string = dom.toprettyxml()

    with open(urdf_file, "w") as f:
        f.write(pretty_xml_as_string)

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

if __name__ == '__main__':

    ## As a first step, we locate the URDF file and parse it
    parser = argparse.ArgumentParser(description="Process a URDF file to merge fixed joints and generate header file.")
    parser.add_argument('urdf_file', type=str, nargs='?', default='./models/RA830/modified_ra830.urdf', help='Path to the URDF file')
    args = parser.parse_args()
    
    urdf_file = args.urdf_file
    formatted_urdf = format_urdf(urdf_file)
    print_element(formatted_urdf)

    # Write the formatted URDF to a new file
    formatted_urdf_file = urdf_file.replace('.urdf', '_formatted.urdf')
    tree = ET.ElementTree(formatted_urdf)
    dom = xml.dom.minidom.parseString(ET.tostring(tree.getroot()))
    pretty_xml_as_string = dom.toprettyxml()

    with open(formatted_urdf_file, "w") as f:
        f.write(pretty_xml_as_string)

    