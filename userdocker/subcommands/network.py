# -*- coding: utf-8 -*-
import argparse
from functools import partial

from ..helpers.cmd import init_cmd
from ..helpers.execute import exit_exec_cmd
from ..helpers.parser import init_subcommand_parser
from ..config import user_name


def prefixed_string(v, prefix):
    if not v.startswith(prefix):
        raise argparse.ArgumentTypeError(
            "value must begin with user name %r" % prefix
        )
    return v


def _network_argument(parser, prefix=user_name+"_"):
    parser.add_argument(
        "network",
        type=partial(prefixed_string, prefix=prefix),
        help="network name, must start with user name %r" % prefix,
    )


def parser_network(parser):
    sub_parser = init_subcommand_parser(parser, "network")
    action_parser = sub_parser.add_subparsers(
        dest="action",
        help="docker network action"
    )

    create_parser = action_parser.add_parser("create")
    _network_argument(create_parser)

    rm_parser = action_parser.add_parser("rm")
    _network_argument(rm_parser)

    action_parser.add_parser("ls")


def exec_cmd_network(args):
    cmd = init_cmd(args)
    create_args = cmd[2:]
    cmd = cmd[:2]

    if args.action in ("create", "rm") and not args.network:
        raise argparse.ArgumentTypeError(
            "the following arguments are required: network",
        )
    if args.action == "create":
        cmd.append(args.action)
        cmd.extend(create_args)
        cmd.append(args.network)
    elif args.action == "rm":
        cmd.append(args.action)
    elif args.action == "ls":
        cmd.append(args.action)

    exit_exec_cmd(cmd, dry_run=args.dry_run)
