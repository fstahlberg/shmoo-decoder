import os
import importlib

def import_classes(classes_dir, namespace):
    for file in os.listdir(classes_dir):
        path = os.path.join(classes_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            class_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + class_name)
