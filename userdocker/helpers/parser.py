# -*- coding: utf-8 -*-

from ..config import ARGS_ALWAYS
from ..config import ARGS_AVAILABLE
from ..config.arguments import Argument


def create_arguments(args, admin_enforced):
    out_args = []
    for spec in args:
        if isinstance(spec, Argument):
            spec.admin_enforced = admin_enforced
        elif isinstance(spec, str):
            # just a single arg as string
            spec = Argument(spec, admin_enforced=admin_enforced)
        elif isinstance(spec, (list, tuple)):
            # aliases as list or tuple
            spec = Argument(*spec, admin_enforced=admin_enforced)
        else:
            raise NotImplementedError(
                "Cannot understand admin defined ARG %s" % spec
            )
        out_args.append(spec)
    return out_args


def init_subcommand_parser(parent_parser, scmd):
    parser = parent_parser.add_parser(
        scmd,
        help='Lets a user run "docker %s ..." command' % scmd,
    )
    parser.set_defaults(
        patch_through_args=[],
    )

    # patch args through
    _args_seen = set()
    try:
        specs = create_arguments(ARGS_AVAILABLE.get(scmd, []), False)
        specs.extend(create_arguments(ARGS_ALWAYS.get(scmd, []), True))
    except NotImplementedError as e:
        raise NotImplementedError("Error parsing config for command %s:" % scmd, *e.args)

    for spec in specs:
        # raise error when duplicate arguments are detected
        if any(arg in _args_seen for arg in spec.arguments):
            raise NotImplementedError(
                "Cannot understand admin defined ARG %s for command %s" % (
                    spec, scmd))
        _args_seen.update(spec.arguments)

        parser.add_argument(*spec.arguments, **spec.kwds)

    return parser
