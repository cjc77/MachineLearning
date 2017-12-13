import os
import imp


def load_src(file_dir, module):
    path = os.path.join(os.path.dirname("__file__"), file_dir)
    return imp.load_source(module + ".py", path)
