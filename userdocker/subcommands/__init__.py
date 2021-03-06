# -*- coding: utf-8 -*-

from .attach import *
from .dockviz import *
from .images import *
from .ps import *
from .pull import *
from .run import *
from .network import *
from .version import *

SPECIFIC_PARSER_PREFIX = 'parser_'
specific_parsers = {
    _var.split(SPECIFIC_PARSER_PREFIX)[1]: _val
    for _var, _val in globals().items()
    if _var.startswith(SPECIFIC_PARSER_PREFIX)
}

SPECIFIC_CMD_EXECUTOR_PREFIX = 'exec_cmd_'
specific_command_executors = {
    _var.split(SPECIFIC_CMD_EXECUTOR_PREFIX)[1]: _val
    for _var, _val in globals().items()
    if _var.startswith(SPECIFIC_CMD_EXECUTOR_PREFIX)
}

__all__ = [specific_parsers, specific_command_executors]
