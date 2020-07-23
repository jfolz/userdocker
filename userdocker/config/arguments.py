# -*- coding: utf-8 -*-

import argparse
from fnmatch import fnmatch


class _PatchThroughAssignmentAction(argparse._AppendAction):
    """
    Action that appends not only the value but the option=value to dest.

    Useful for patch through args with assignment like: --shm-size=1g.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        super(_PatchThroughAssignmentAction, self).__call__(
            parser, namespace,
            values=option_string + '=' + values,
            option_string=option_string,
        )


class Argument:
    """
    An Argument to be used with userdocker configs.
    Use as type with add_argument.
    Does not perform any checks and simply returns the given value.

    Arguments will be appended to patch_through_args as-is.
    """
    def __init__(self, *arguments, admin_enforced=False):
        self.kwds = {
            "dest": "patch_through_args",
        }
        for arg in arguments:
            # make sure each arg starts with - and doesn't contain ' '
            if not arg.startswith('-') or ' ' in arg:
                raise NotImplementedError(
                    "Admin defined ARG must start with '-'"
                    " and my not contain spaces: %s" % arg
                )
            # if ARG has value
            if '=' in arg:
                # make sure there is only one argument with value
                if len(arguments) != 1:
                    raise NotImplementedError(
                        "ARG with value must be single: %s" % arg
                    )
                arg, _, value = arg.partition("=")
                arguments = [arg]
                self.kwds["action"] = _PatchThroughAssignmentAction
                self.kwds["choices"] = [value]
            # ARG is value-less, just append ARG as const
            else:
                self.kwds["action"] = "append_const"
                self.kwds["const"] = arguments[0]

        self.arguments = arguments
        self._admin_enforced = None
        self.admin_enforced = admin_enforced

    @property
    def admin_enforced(self):
        return self._admin_enforced

    @admin_enforced.setter
    def admin_enforced(self, value):
        self._admin_enforced = value
        self.kwds["help"] = "see docker help" + (
            " (enforced by admin)" if value else ""
        )

    def __eq__(self, other):
        if isinstance(other, Argument):
            return any(arg in self.arguments for arg in other.arguments)
        elif isinstance(other, str):
            return other in self.arguments
        else:
            return False

    def __call__(self, value):
        return value


class GlobArgument(Argument):
    """
    Argument where values must match a given GLOB pattern.
    Raises ArgumentTypeError if value does not match pattern.
    """
    def __init__(self, pattern, *arguments, admin_enforced=False):
        super(GlobArgument, self).__init__(
            *arguments,
            admin_enforced=admin_enforced
        )
        self.pattern = pattern
        self.kwds["action"] = _PatchThroughAssignmentAction
        self.kwds["type"] = self
        self.kwds.pop("const")

    def __hash__(self):
        return hash(self.arguments) + hash(self.pattern)

    def __call__(self, value):
        if fnmatch(value, self.pattern):
            return value
        raise argparse.ArgumentTypeError(
            "value must match pattern %r" % self.pattern
        )
