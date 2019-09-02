"""https://github.com/NiklasRosenstein/pydoc-markdown/blob/master/pydocmd/__main__.py"""

import inspect
import os
import sys
import yaml

from pydocmd.document import Index
from pydocmd.imp import dir_object
from pydocmd.loader import PythonLoader
from pydocmd.preprocessor import Preprocessor


def callable_to_source_link(obj):
    repo = "larq/larq"
    obj = inspect.unwrap(obj)
    path = inspect.getfile(obj).lstrip("./")
    if "larq_zoo" in path:
        path = path[path.find("larq_zoo") :]
        repo = "larq/zoo"
    source = inspect.getsourcelines(obj)
    line = source[-1] + 1 if source[0][0].startswith("@") else source[-1]
    link = f"https://github.com/{repo}/blob/master/{path}#L{line}"
    return f'<a class="headerlink code-link" style="float:right;" href="{link}" title="Source code"></a>'


class PythonLoaderWithSource(PythonLoader):
    def load_section(self, section):
        super().load_section(section)
        obj = section.loader_context["obj"]
        if callable(obj):
            section.title += callable_to_source_link(obj)


with open("apidocs.yml", "r") as stream:
    api_structure = yaml.safe_load(stream)

# Build the index and document structure first, we load the actual
# docstrings at a later point.
print("Building index...")
index = Index()


def add_sections(doc, object_names, depth=1):
    if isinstance(object_names, list):
        [add_sections(doc, x, depth) for x in object_names]
    elif isinstance(object_names, dict):
        for key, subsections in object_names.items():
            add_sections(doc, key, depth)
            add_sections(doc, subsections, depth + 1)
    elif isinstance(object_names, str):
        # Check how many levels of recursion we should be going.
        expand_depth = len(object_names)
        object_names = object_names.rstrip("+")
        expand_depth -= len(object_names)

        def create_sections(name, level):
            if level > expand_depth:
                return
            index.new_section(doc, name, depth=depth + level, header_type="markdown")
            for sub in dir_object(name, "line", False):
                sub = name + "." + sub
                create_sections(sub, level + 1)

        create_sections(object_names, 0)
    else:
        raise RuntimeError(object_names)


# Make sure that we can find modules from the current working directory,
# and have them take precedence over installed modules.
sys.path.insert(0, ".")

for pages in api_structure:
    for fname, object_names in pages.items():
        doc = index.new_document(fname)
        add_sections(doc, object_names)

loader = PythonLoaderWithSource({})
preproc = Preprocessor({})

preproc.link_lookup = {}
for file, doc in index.documents.items():
    for section in doc.sections:
        preproc.link_lookup[section.identifier] = file
# Load the docstrings and fill the sections.
print("Started generating documentation...")
for doc in index.documents.values():
    for section in filter(lambda s: s.identifier, doc.sections):
        loader.load_section(section)
        preproc.preprocess_section(section)

# Write out all the generated documents.
os.makedirs(os.path.join("docs", "api"), exist_ok=True)
for fname, doc in index.documents.items():
    with open(os.path.join("docs", "api", fname), "w") as fp:
        for section in doc.sections:
            section.render(fp)
