# -*- coding: utf-8 -*-

import argparse
import logging
import os
import re
import ipaddress
import time
import signal

from .. import __version__
from ..config import EXECUTORS
from ..config import ALLOWED_IMAGE_REGEXPS
from ..config import ALLOWED_PORT_MAPPINGS
from ..config import CAPS_ADD
from ..config import CAPS_DROP
from ..config import ENV_VARS
from ..config import ENV_VARS_EXT
from ..config import NV_ALLOW_OWN_GPU_REUSE
from ..config import NV_ALLOWED_GPUS
from ..config import NV_DEFAULT_GPU_COUNT_RESERVATION
from ..config import NV_MAX_GPU_COUNT_RESERVATION
from ..config import NV_USE_CUDA_VISIBLE_DEVICES
from ..config import SLURM_BIND_GPU
from ..config import SLURM_NETWORK_ADDRESS_OFFSET
from ..config import PROBE_USED_MOUNTS
from ..config import RUN_PULL
from ..config import USER_IN_CONTAINER
from ..config import VOLUME_MOUNTS_ALWAYS
from ..config import VOLUME_MOUNTS_AVAILABLE
from ..config import VOLUME_MOUNTS_DEFAULT
from ..config import ADDITIONAL_ARGS
from ..config import gid
from ..config import uid
from ..config import user_name
from ..helpers.cmd import init_cmd
from ..helpers.exceptions import UserDockerException
from ..helpers.execute import exec_cmd
from ..helpers.execute import exit_exec_cmd
from ..helpers.logger import logger
from ..helpers.nvidia import nvidia_get_available_gpus
from ..helpers.parser import init_subcommand_parser
from .network import prefixed_string


def parser_run(parser):
    sub_parser = init_subcommand_parser(parser, 'run')

    sub_parser.add_argument(
        "--no-default-mounts",
        help="does not automatically add default mounts",
        action="store_true",
    )

    mounts_help = []
    if VOLUME_MOUNTS_ALWAYS:
        mounts_help += ['Admin enforced: %s.' % ', '.join(VOLUME_MOUNTS_ALWAYS)]
    if VOLUME_MOUNTS_DEFAULT:
        mounts_help += ['Default: %s.' % ', '.join(VOLUME_MOUNTS_DEFAULT)]
    if VOLUME_MOUNTS_AVAILABLE:
        mounts_help += ['Available: %s.' % ', '.join(VOLUME_MOUNTS_AVAILABLE)]
    if mounts_help:
        sub_parser.add_argument(
            "-v", "--volume",
            help="user specified volume mounts (can be given multiple times). "
                 "%s" % " ".join(mounts_help),
            action="append",
            dest="volumes",
            default=[],
        )

    sub_parser.add_argument(
        "--entrypoint",
        help="Overwrite the default ENTRYPOINT of the image",
    )

    sub_parser.add_argument(
        "-w", "--workdir",
        help="Working directory inside the container",
    )

    if ALLOWED_PORT_MAPPINGS:
        sub_parser.add_argument(
            "-p", "--publish",
            help="Publish a container's ports to the host (see docker help). "
                 "Allowed: " + ', '.join(ALLOWED_PORT_MAPPINGS),
            action="append",
            dest="port_mappings",
            default=[],
        )

    sub_parser.add_argument(
        "image",
        help="the image to run. Allowed: " + ', '.join(ALLOWED_IMAGE_REGEXPS),
    )

    sub_parser.add_argument(
        "image_args",
        help="arguments passed to the image",
        nargs=argparse.REMAINDER
    )


__canary = object()


def getenv_raise(key, default=__canary, msg='{} environment variable is not set'):
    v = os.getenv(key, default)
    if v is __canary:
        raise UserDockerException(msg.format(key))
    return v


def is_slurm_job():
    return 'SLURM_JOBID' in os.environ and 'SLURM_TASK_PID' in os.environ


def prepare_nvidia_docker_run(args):
    # mainly handles GPU arbitration via ENV var for nvidia-docker
    # note that these are ENV vars for the command, not the container

    if os.getenv('NV_HOST'):
        raise UserDockerException('ERROR: NV_HOST env var not supported yet')

    # check if allowed
    if not NV_ALLOWED_GPUS:
        raise UserDockerException(
            "ERROR: No GPUs available due to admin setting."
        )

    # depending on config try CUDA_VISIBLE_DEVICES first (used by slurm)
    nv_gpus = ''
    if NV_USE_CUDA_VISIBLE_DEVICES:
        nv_gpus = os.getenv('CUDA_VISIBLE_DEVICES', nv_gpus)
    # then NV_GPU (normal nvidia-docker env var)
    nv_gpus = os.getenv('NV_GPU', nv_gpus)
    if nv_gpus:
        # the user has set NV_GPU, just check if it's ok
        nv_gpus = [g.strip() for g in nv_gpus.split(',')]
        try:
            nv_gpus = [int(gpu) for gpu in nv_gpus]
        except ValueError as e:
            raise UserDockerException(
                "ERROR: Can't parse NV_GPU, use index notation: %s" % e
            )

        if not (
                NV_ALLOWED_GPUS == 'ALL'
                or all(gpu in NV_ALLOWED_GPUS for gpu in nv_gpus)):
            raise UserDockerException(
                "ERROR: Access to at least one specified NV_GPU denied by "
                "admin. Available GPUs: %r" % (NV_ALLOWED_GPUS,)
            )

        # check if in bounds (and MAX >= 0)
        if 0 <= NV_MAX_GPU_COUNT_RESERVATION < len(nv_gpus):
            raise UserDockerException(
                "ERROR: Number of requested GPUs > %d (admin limit)" % (
                    NV_MAX_GPU_COUNT_RESERVATION,)
            )

        # check if available
        gpus_available, own_gpus = nvidia_get_available_gpus(args.executor_path)
        if NV_ALLOW_OWN_GPU_REUSE:
            gpus_available.extend(own_gpus)
        for g in nv_gpus:
            if g not in gpus_available:
                msg = (
                    'ERROR: GPU %d is currently not available!\nUse:\n'
                    '"sudo userdocker ps --gpu-free" to find available GPUs.\n'
                    '"sudo userdocker ps --gpu-used" and "nvidia-smi" to see '
                    'status.' % g
                )
                if NV_ALLOW_OWN_GPU_REUSE and own_gpus:
                    msg += '\n"sudo userdocker ps --gpu-used-mine to show own' \
                           '(reusable) GPUs.'
                raise UserDockerException(msg)
    else:
        # NV_GPU wasn't set, use admin defaults, tell user
        gpu_default = NV_DEFAULT_GPU_COUNT_RESERVATION
        logger.info(
            "NV_GPU environment variable not set, trying to acquire admin "
            "default of %d GPUs" % gpu_default
        )
        gpus_available, own_gpus = nvidia_get_available_gpus(args.executor_path)
        nv_gpus = gpus_available[:gpu_default]
        if len(nv_gpus) < gpu_default:
            msg = (
                'Could not find %d available GPU(s)!\nUse:\n'
                '"sudo userdocker ps --gpu-used" and "nvidia-smi" to see '
                'status.' % gpu_default
            )
            if NV_ALLOW_OWN_GPU_REUSE and own_gpus:
                msg += '\n You can set NV_GPU to reuse a GPU you have already' \
                       ' reserved.'
            raise UserDockerException(msg)
        gpu_env = ",".join([str(g) for g in nv_gpus])
        logger.info("Setting NV_GPU=%s" % gpu_env)

    # for slurm jobs distribute nv_gpus to tasks on this node
    if SLURM_BIND_GPU and is_slurm_job():
        # TODO check SLURM_LOCALID for single task jobs
        local_id = int(getenv_raise('SLURM_LOCALID', 0))
        # TODO check SLURM_NTASKS_PER_NODE for single task jobs
        tasks = int(getenv_raise('SLURM_NTASKS_PER_NODE', 1))
        step = tasks / len(nv_gpus)
        if step % 1 != 0:
            logger.warning('cannot distribute %d GPUs evenly to %d tasks'
                           % (len(nv_gpus), tasks))
        first = int(round(local_id*step))
        last = int(round((local_id+1)*step))
        nv_gpus = nv_gpus[first:last]

    gpu_env = ",".join([str(g) for g in nv_gpus])
    os.environ['NV_GPU'] = gpu_env


def ip_address(subnet, procid):
    # TODO IPv6
    net = subnet.partition('/')[0].split('.')
    addr = list(map(int, net))[::-1]
    available_addresses = 255 - SLURM_NETWORK_ADDRESS_OFFSET
    addr[0] = (procid + SLURM_NETWORK_ADDRESS_OFFSET) % available_addresses
    rem = (procid + SLURM_NETWORK_ADDRESS_OFFSET) // available_addresses
    for i in range(1, len(addr)):
        available_addresses = 256 - addr[i]
        addr[i] += rem % available_addresses
        rem //= available_addresses
    addr = '.'.join(map(str, addr[::-1]))
    if ipaddress.ip_address(addr) not in ipaddress.ip_network(subnet):
        raise UserDockerException(
            'Task %d address %s is not in subnet %s. Too many tasks for subnet?'
            % (procid, addr, subnet)
        )
    return addr


def set_network_ip_address():
    cmd = []
    network = getenv_raise('USERDOCKER_NETWORK_NAME', None)
    if network is not None:
        if prefixed_string(network) != network:
            raise UserDockerException()
        subnet = getenv_raise('USERDOCKER_NETWORK_SUBNET')
        procid = int(getenv_raise('SLURM_PROCID'))
        cmd += [
            '-e', 'USERDOCKER_RANK0_ADDRESS=%s' % ip_address(subnet, 0),
            '--network', network,
            '--ip', ip_address(subnet, procid),
        ]
    return cmd


def container_name():
    jobid = getenv_raise('SLURM_JOBID', str(time.time())[-4:])
    procid = getenv_raise('SLURM_PROCID', str(os.getpid()))
    name = '%s_%s_%s' % (user_name, jobid, procid)
    os.environ["USERDOCKER_CONTAINER_NAME"] = name
    return ['--name', name]


def handle_sigterm_docker_stop(*_, **__):
    cmd = [
        EXECUTORS["docker"],
        "stop",
        os.environ["USERDOCKER_CONTAINER_NAME"]
    ]
    exec_cmd(cmd)


def exec_cmd_run(args):
    cmd = init_cmd(args)

    # container name
    cmd += container_name()

    # add additional args first
    cmd.extend(ADDITIONAL_ARGS)

    # check port mappings
    for pm in getattr(args, 'port_mappings', []):
        for pm_pattern in ALLOWED_PORT_MAPPINGS:
            if re.match(pm_pattern, pm):
                cmd += ['-p', pm]
                break
        else:
            raise UserDockerException(
                "ERROR: given port mapping not allowed: %s" % pm
            )

    # check mounts
    mounts = []
    mounts_available = \
        VOLUME_MOUNTS_ALWAYS + VOLUME_MOUNTS_DEFAULT + VOLUME_MOUNTS_AVAILABLE

    mounts += VOLUME_MOUNTS_ALWAYS

    if not args.no_default_mounts:
        mounts += VOLUME_MOUNTS_DEFAULT

    for user_mount in getattr(args, 'volumes', []):
        if user_mount in mounts:
            continue
        if user_mount in mounts_available:
            mounts += [user_mount]
            continue

        # literal matches didn't work, check if the user appended a 'ro' flag
        if len(user_mount.split(':')) == 3:
            host_path, container_path, flag = user_mount.split(':')
            if flag == 'ro':
                st = ':'.join([host_path, container_path])
                if st in mounts:
                    # upgrade mount to include ro flag
                    idx = mounts.index(st)
                    mounts[idx] = user_mount
                    continue
                if st in mounts_available:
                    mounts += [user_mount]
                    continue

        # allow potential unspecified container_path mounts
        host_path = user_mount.split(':')[0] + ':'
        if host_path in mounts_available:
            mounts += [user_mount]
            continue

        raise UserDockerException(
            "ERROR: given mount not allowed: %s" % user_mount
        )

    mount_host_paths = [m.split(':')[0] for m in mounts]
    for ms in mount_host_paths:
        if not os.path.exists(ms):
            raise UserDockerException(
                "ERROR: mount can't be found: %s" % ms
            )
        if PROBE_USED_MOUNTS and os.path.isdir(ms):
            os.listdir(ms)

    for mount in mounts:
        if ':' not in mount:
            raise UserDockerException(
                "ERROR: anonymous mounts currently not supported: %s" % mount
            )
        cmd += ["-v", mount]

    # setup environment with nvidia-specific options
    if args.executor == 'nvidia-docker':
        prepare_nvidia_docker_run(args)

    # userdocker environment
    env_vars = ENV_VARS + ENV_VARS_EXT.get(args.executor, [])
    env_vars += [
        "USERDOCKER=%s" % __version__,
        "USERDOCKER_USER=%s" % user_name,
        "USERDOCKER_UID=%d" % uid,
    ]
    if args.executor == 'nvidia-docker':
        # remember which GPU was assigned to the container for ps --gpu-used
        env_vars += [
            "USERDOCKER_NV_GPU=%s" % os.environ['NV_GPU']
        ]
    for env_var in env_vars:
        cmd += ['-e', env_var]

    # slurm env vars & communication
    if is_slurm_job():
        procid = int(getenv_raise('SLURM_PROCID'))
        nnodes = int(getenv_raise('SLURM_NNODES'))
        ntasks = int(getenv_raise('SLURM_NTASKS'))
        # IP address and env vars for process
        cmd += [
            '-e', 'SLURM_PROCID=%d' % procid,
            '-e', 'SLURM_NNODES=%d' % nnodes,
            '-e', 'SLURM_NTASKS=%d' % ntasks,
        ]

    # if network arg is defined, add ip address
    cmd += set_network_ip_address()

    # set user inside container
    if USER_IN_CONTAINER:
        cmd += ["-u", "%d:%d" % (uid, gid)]

    for cap_drop in CAPS_DROP:
        cmd += ["--cap-drop=%s" % cap_drop]
    for cap_add in CAPS_ADD:
        cmd += ["--cap-add=%s" % cap_add]

    if args.workdir:
        cmd += ["-w", args.workdir]
    if args.entrypoint:
        cmd += ["--entrypoint", args.entrypoint]

    # additional injection protection, deactivated for now due to nvidia-docker
    # unability to handle this
    # cmd.append("--")

    img = args.image
    if ":" not in img and "@" not in img:
        # user didn't explicitly set a tag or digest, append ":latest"
        img += ":latest"

    if ALLOWED_IMAGE_REGEXPS:
        for air in ALLOWED_IMAGE_REGEXPS:
            if re.match(air, img):
                break
        else:
            raise UserDockerException(
                "ERROR: image %s not in allowed image regexps: %s" % (
                    img, ALLOWED_IMAGE_REGEXPS))

    # pull image?
    if RUN_PULL == "default":
        # just let `docker run` do its thing
        pass
    elif RUN_PULL == "always":
        # pull image
        exec_cmd(
            [args.executor_path, 'pull', img],
            dry_run=args.dry_run,
            loglvl=logging.DEBUG,
        )
    elif RUN_PULL == "never":
        # check if image is available locally
        tmp = exec_cmd(
            [args.executor_path, 'images', '-q', img],
            return_status=False,
            loglvl=logging.DEBUG,
        )
        if not tmp:
            raise UserDockerException(
                "ERROR: you can only use locally available images, but %s could"
                " not be found locally" % img
            )
    else:
        raise UserDockerException(
            "ERROR: RUN_PULL config variable not expected range, contact admin"
        )

    cmd.append(img)
    cmd.extend(args.image_args)

    # install the SIGTERM handler to stop the container
    signal.signal(signal.SIGTERM, handle_sigterm_docker_stop)

    exit_exec_cmd(cmd, dry_run=args.dry_run)
