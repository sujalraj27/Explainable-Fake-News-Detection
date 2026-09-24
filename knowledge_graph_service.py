import spacy
import networkx as nx
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

# ----------------------------
# Load SpaCy Model
# ----------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "SpaCy model not found.\n"
        "Run:\n"
        "python -m spacy download en_core_web_sm"
    )


# ----------------------------
# Entity & Relation Extraction
# ----------------------------

def extract_entities_and_relations(text):

    doc = nlp(text)

    triplets = []

    # -------- Subject Verb Object Extraction --------

    for sentence in doc.sents:

        subject = None
        relation = None
        obj = None

        for token in sentence:

            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text

            elif token.pos_ == "VERB":
                relation = token.lemma_

            elif token.dep_ in ("dobj", "pobj", "attr"):
                obj = token.text

        if subject and relation and obj:

            triplets.append(

                (
                    subject,
                    relation,
                    obj
                )

            )

    # -------- Named Entity Fallback --------

    if len(triplets) == 0:

        nodes = []

        # Named Entities

        for ent in doc.ents:

            if ent.text not in nodes:
                nodes.append(ent.text)

        # Noun Chunks

        for chunk in doc.noun_chunks:

            if chunk.text not in nodes:
                nodes.append(chunk.text)

        # Sequential Connections

        for i in range(len(nodes) - 1):

            triplets.append(

                (
                    nodes[i],
                    "related_to",
                    nodes[i + 1]
                )

            )

    return triplets


# ----------------------------
# Knowledge Graph Generator
# ----------------------------

def generate_knowledge_graph(text):

    if not text.strip():

        return {

            "nodes": [],

            "edges": [],

            "graph_image": ""

        }

    triplets = extract_entities_and_relations(text)

    G = nx.DiGraph()

    nodes = set()

    edges = []

    # ----------------------------

    for source, relation, target in triplets:

        G.add_edge(

            source,

            target,

            label=relation

        )

        nodes.add(source)

        nodes.add(target)

        edges.append(

            {

                "source": source,

                "target": target,

                "relation": relation

            }

        )

    # ----------------------------
    # Draw Graph
    # ----------------------------

    plt.figure(figsize=(10, 7))

    pos = nx.spring_layout(

        G,

        seed=42,

        k=1.5

    )

    nx.draw_networkx_nodes(

        G,

        pos,

        node_color="#4A90E2",

        node_size=2600,

        alpha=0.95

    )

    nx.draw_networkx_edges(

        G,

        pos,

        edge_color="gray",

        width=2,

        arrows=True,

        arrowsize=20,

        arrowstyle="-|>"

    )

    nx.draw_networkx_labels(

        G,

        pos,

        font_size=9,

        font_color="white",

        font_weight="bold"

    )

    edge_labels = nx.get_edge_attributes(

        G,

        "label"

    )

    nx.draw_networkx_edge_labels(

        G,

        pos,

        edge_labels=edge_labels,

        font_size=8,

        font_color="red"

    )

    plt.title(

        "Knowledge Graph",

        fontsize=14,

        fontweight="bold"

    )

    plt.axis("off")

    # ----------------------------
    # Save Image
    # ----------------------------

    buffer = io.BytesIO()

    plt.savefig(

        buffer,

        format="png",

        dpi=300,

        bbox_inches="tight"

    )

    buffer.seek(0)

    plt.close()

    image_base64 = base64.b64encode(

        buffer.getvalue()

    ).decode()

    return {

        "nodes": list(nodes),

        "edges": edges,

        "graph_image": "data:image/png;base64," + image_base64

    }