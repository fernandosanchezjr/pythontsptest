# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:18:21 2018

@author: fernando
"""
import logging
from os import path


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(logging.StreamHandler())


def get_relative_path(module: str, path_name: str) -> str:
    return path.abspath(path.join(path.dirname(__file__), path_name))


setup_logging()
