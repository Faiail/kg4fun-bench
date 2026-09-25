import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

from src.utils import load_json, ParameterKeys
from src.preprocess.ontomap.kg4fun_fields import ClassFields, RelFields


class LogMapPreprocessor:
    """
    Reads a raw split directory (e.g. data/raw/kg4fun/dataset1/splits/hybrid/test)
    and generates the standard MELT track structure:
    - source.rdf (OWL/XML format)
    - target.rdf (OWL/XML format)
    - reference.rdf (OAEI Alignment API XML format)
    """

    def __init__(self, parameters: dict):
        self.parameters = parameters
        self._init_paths()

    def _init_paths(self):
        general_parameters = self.parameters.get(ParameterKeys.GENERAL, dict())
        self.data_dir = general_parameters.get(ParameterKeys.DATA_DIR, "")
        self.out_dir = general_parameters.get(ParameterKeys.OUT_DIR, "")
        self.pbar = general_parameters.get(ParameterKeys.PBAR, True)
        os.makedirs(self.out_dir, exist_ok=True)

    def _prettify_xml(self, elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def _create_ontology(self, title, uri_base, classes, properties):
        """Creates an OWL/XML Ontology using ElementTree."""
        rdf_ns = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        owl_ns = "http://www.w3.org/2002/07/owl#"
        rdfs_ns = "http://www.w3.org/2000/01/rdf-schema#"

        ET.register_namespace('rdf', rdf_ns)
        ET.register_namespace('owl', owl_ns)
        ET.register_namespace('rdfs', rdfs_ns)

        root = ET.Element(f"{{{rdf_ns}}}RDF")
        root.set(f"xml:base", uri_base)

        onto = ET.SubElement(root, f"{{{owl_ns}}}Ontology")
        onto.set(f"{{{rdf_ns}}}about", uri_base)

        for cls_id, cls_info in classes.items():
            cls_elem = ET.SubElement(root, f"{{{owl_ns}}}Class")
            cls_elem.set(f"{{{rdf_ns}}}about", f"{uri_base}#{cls_id}")

            label = cls_info.get("label", "")
            if label:
                lbl_elem = ET.SubElement(cls_elem, f"{{{rdfs_ns}}}label")
                lbl_elem.text = label

            comment = cls_info.get("comment", "")
            if comment and comment != "-":
                cmt_elem = ET.SubElement(cls_elem, f"{{{rdfs_ns}}}comment")
                cmt_elem.text = comment

        for prop_id, prop_info in properties.items():
            prop_elem = ET.SubElement(root, f"{{{owl_ns}}}ObjectProperty")
            prop_elem.set(f"{{{rdf_ns}}}about", f"{uri_base}#{prop_id}")

            label = prop_info.get("label", "")
            if label:
                lbl_elem = ET.SubElement(prop_elem, f"{{{rdfs_ns}}}label")
                lbl_elem.text = label

            comment = prop_info.get("comment", "")
            if comment and comment != "-":
                cmt_elem = ET.SubElement(prop_elem, f"{{{rdfs_ns}}}comment")
                cmt_elem.text = comment

        return root

    def _create_alignment_xml(self, source_base, target_base, mappings):
        """Creates an OAEI Alignment API XML using ElementTree."""
        align_ns = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
        rdf_ns = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xsd_ns = "http://www.w3.org/2001/XMLSchema#"

        ET.register_namespace('', align_ns)
        ET.register_namespace('rdf', rdf_ns)
        ET.register_namespace('xsd', xsd_ns)
        
        root = ET.Element(f"{{{rdf_ns}}}RDF")
        
        alignment = ET.SubElement(root, f"{{{align_ns}}}Alignment")
        
        xml_elem = ET.SubElement(alignment, f"{{{align_ns}}}xml")
        xml_elem.text = "yes"
        level_elem = ET.SubElement(alignment, f"{{{align_ns}}}level")
        level_elem.text = "0"
        type_elem = ET.SubElement(alignment, f"{{{align_ns}}}type")
        type_elem.text = "**"
        
        for src_id, tgt_id in mappings:
            map_elem = ET.SubElement(alignment, f"{{{align_ns}}}map")
            cell_elem = ET.SubElement(map_elem, f"{{{align_ns}}}Cell")
            
            e1 = ET.SubElement(cell_elem, f"{{{align_ns}}}entity1")
            e1.set(f"{{{rdf_ns}}}resource", f"{source_base}#{src_id}")
            
            e2 = ET.SubElement(cell_elem, f"{{{align_ns}}}entity2")
            e2.set(f"{{{rdf_ns}}}resource", f"{target_base}#{tgt_id}")
            
            relation = ET.SubElement(cell_elem, f"{{{align_ns}}}relation")
            relation.text = "="
            
            measure = ET.SubElement(cell_elem, f"{{{align_ns}}}measure")
            measure.set(f"{{{rdf_ns}}}datatype", f"{xsd_ns}float")
            measure.text = "1.0"
            
        return root

    def __call__(self):
        print(f"Reading raw split data from {self.data_dir}...")

        # 1. Load Node Info (Source Instances & Target Classes)
        node_info = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODE_INFO}.json"))
        node_type_info = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODE_TYPE_INFO}.json"))
        
        # 2. Load Edge Info (Source Relations & Target Relations)
        edge_info = load_json(os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_INFO}.json"))
        edge_types_list = load_json(os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_TYPES}.json"))
        
        # 3. Load Alignments
        nodes = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODES}.json"))
        edge_mappings = load_json(os.path.join(self.data_dir, ParameterKeys.EDGES, f"{ParameterKeys.EDGE_COMPONENT_MAPPING}.json"))

        # Build Source Dictionary (Instances + Edges)
        source_classes = {}
        for n in nodes:
            qid = n[ParameterKeys.QID]
            info = node_info.get(qid, {})
            source_classes[qid] = {
                "label": info.get("itemLabel", ""),
                "comment": info.get("itemDescription", "")
            }

        source_properties = {}
        for m in edge_mappings:
            pid = m[ParameterKeys.EDGE_TYPE][1]
            info = edge_info.get(pid, {})
            source_properties[pid] = {
                "label": info.get("itemLabel", ""),
                "comment": info.get("itemDescription", "")
            }

        # Build Target Dictionary (Classes + Edges)
        node_types = load_json(os.path.join(self.data_dir, f"{ParameterKeys.NODE_TYPES}.json"))
        
        target_classes = {}
        target_properties = {}
        
        qid_to_cls = {}
        for schema_qid, info in node_types.items():
            cls_idx = info[ParameterKeys.CLS_IDX]
            cls_id = f"cls_{cls_idx}"
            qid_to_cls[schema_qid] = cls_id
            
            t_info = node_type_info.get(schema_qid, {})
            target_classes[cls_id] = {
                "label": t_info.get("itemLabel", ""),
                "comment": t_info.get("itemDescription", "")
            }

        pid_to_rel = {}
        for idx, et in enumerate(edge_types_list):
            pid = et[ParameterKeys.PID]
            rel_id = f"rel_{idx}"
            pid_to_rel[pid] = rel_id
            
            t_info = edge_info.get(pid, {})
            target_properties[rel_id] = {
                "label": t_info.get("itemLabel", ""),
                "comment": t_info.get("itemDescription", "")
            }

        # Build Mappings (Ground Truth)
        mappings = []
        
        # Node mappings
        for n in nodes:
            src_qid = n[ParameterKeys.QID]
            tgt_cls_idx = n[ParameterKeys.CLS_IDX]
            mappings.append((src_qid, f"cls_{tgt_cls_idx}"))
            
        # Edge mappings
        for m in edge_mappings:
            src_pid = m[ParameterKeys.EDGE_TYPE][1]
            tgt_comp_id = m.get("component_id", None)
            
            if tgt_comp_id is not None:
                mappings.append((src_pid, f"rel_{tgt_comp_id}"))

        # Generate XMLs
        source_base = "http://kg4fun.org/source"
        target_base = "http://kg4fun.org/target"

        print("Generating source.rdf...")
        source_xml = self._create_ontology("Source", source_base, source_classes, source_properties)
        with open(os.path.join(self.out_dir, "source.rdf"), "w", encoding="utf-8") as f:
            f.write(self._prettify_xml(source_xml))

        print("Generating target.rdf...")
        target_xml = self._create_ontology("Target", target_base, target_classes, target_properties)
        with open(os.path.join(self.out_dir, "target.rdf"), "w", encoding="utf-8") as f:
            f.write(self._prettify_xml(target_xml))

        print("Generating reference.rdf...")
        align_xml = self._create_alignment_xml(source_base, target_base, mappings)
        with open(os.path.join(self.out_dir, "reference.rdf"), "w", encoding="utf-8") as f:
            f.write(self._prettify_xml(align_xml))

        print(f"Successfully generated MELT track files at {self.out_dir}")
