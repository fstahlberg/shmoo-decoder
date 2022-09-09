import os
import importlib


def import_classes(classes_dir: str, namespace: str):
    """Imports all Python modules in a directory.

    Args:
        classes_dir: Path to the directory with *.py files to import
        namespace: Base namespace. Will be prepended to module names
    """
    for file in os.listdir(classes_dir):
        path = os.path.join(classes_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            cls_name = file[:file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + cls_name)
