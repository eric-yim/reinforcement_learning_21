import os


def whole_path(some_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path,some_name)

def whole_paths(some_list):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i,word in enumerate(some_list):
        some_list[i] = os.path.join(dir_path,word)
    return some_list

def whole_path_connect(base_path,join_path):
    base_path = whole_path(base_path)
    return os.path.join(base_path,join_path)