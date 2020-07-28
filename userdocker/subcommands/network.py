# -*- coding: utf-8 -*-
import argparse
from functools import partial
import json
import sys

from ..helpers.cmd import init_cmd
from ..helpers.execute import exec_cmd
from ..helpers.execute import exit_exec_cmd
from ..helpers.parser import init_subcommand_parser
from ..helpers.exceptions import UserDockerException
from ..config import user_name
from ..config import EXECUTORS


def prefixed_string(v, prefix=user_name+"_"):
    if not v.startswith(prefix):
        raise argparse.ArgumentTypeError(
            "network name must begin with user name %r" % prefix
        )
    return v


def _network_argument(parser, prefix=user_name+"_"):
    parser.add_argument(
        "network",
        type=partial(prefixed_string, prefix=prefix),
        help="network name, must start with %r" % prefix,
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

    inspect_parser = action_parser.add_parser("inspect")
    _network_argument(inspect_parser)

    env_parser = action_parser.add_parser("env")
    _network_argument(env_parser)

    action_parser.add_parser("ls")


def inspect_network(network):
    cmd = [EXECUTORS["docker"], "network", "inspect", network]
    jsontext = exec_cmd(cmd, return_status=False)
    for details in json.loads(jsontext):
        if details["Name"] == network:
            return details


def exec_cmd_network(args):
    args.executor = "docker"
    args.executor_path = EXECUTORS["docker"]
    cmd = init_cmd(args)
    create_args = cmd[2:]
    cmd = cmd[:2]
    cmd.append(args.action)

    if args.action == "create":
        cmd.extend(create_args)
    if args.action in ("create", "rm", "inspect"):
        cmd.append(args.network)
    if args.action == "env":
        details = inspect_network(args.network)
        if details is None:
            raise UserDockerException("No such network: %s" % args.network)
        subnet = details["IPAM"]["Config"][0]["Subnet"]
        print("USERDOCKER_NETWORK_NAME=%s" % args.network,
              "USERDOCKER_NETWORK_SUBNET=%s" % subnet)
        sys.exit(0)

    exit_exec_cmd(cmd, dry_run=args.dry_run)
