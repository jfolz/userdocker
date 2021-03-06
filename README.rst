Userdocker
==========

Userdocker is a wrapper that allows admins to grant restricted docker
commandline access to users.

.. note::

    Userdocker is currently in BETA state. Despite our ongoing efforts to test
    on our local infrastructure, further testing, reviewing and feedback are
    very welcome. Use with caution and watch the GitHub repo for issues and
    new releases!


Userdocker is aimed towards scientific high performance computing and cluster
setups, as they exist in most universities or research groups. Often, such
scientific computations have peculiar dependencies that are difficult to satisfy
across linux distributions (and drive admins crazy ;) ).

In theory such use-cases could largely benefit from docker, as it would allow
users to easily define environments themselves and run them basically without
negative performance impact, as they run directly on the host's kernel. In
reality however granting docker commandline access to users effectively makes
them root equivalent on the host (root in container, volume mount...), making
this prohibitive for cluster computing.

Userdocker solves this problem by wrapping the docker command and just making
the safe parts available to users. Admins can decide what they consider safe
(with sane defaults). The userdocker command largely follows the docker
commandline syntax, so users can use it as an in-place replacement for the
docker command.

Feedback / bugreports / contributions welcome:

https://github.com/joernhees/userdocker


Sample Usage:
=============

.. code-block:: bash

    # command line help (including subcommands the user is allowed to execute)
    sudo userdocker -h

    # (docker images) list images (and useful tree visualization)
    sudo userdocker images
    sudo userdocker dockviz

    # (docker run) run a debian image with user (read-only) mounted home
    sudo userdocker run -it --rm -v $HOME:$HOME:ro debian bash

    # (docker attach) re-attach to own container after connection loss
    sudo userdocker attach 438c7648e76b

    # (docker ps) list running containers
    sudo userdocker ps

    # (docker pull / load) pull or load
    sudo userdocker pull debian
    sudo userdocker load < image.tar.gz

    # (nvidia-docker) extensions for nvidia GPU support
    alias nvidia-userdocker='userdocker --executor=nvidia-docker'
    NV_GPU=1,3,7 nvidia-userdocker run -it --rm nvcr.io/nvidia/tensorflow
    userdocker ps --gpu-used
    userdocker ps --gpu-free

Features:
=========

- Similar commandline interface as ``docker ...`` called ``userdocker ...``
- Support for several docker commands / plugins (docker, nvidia-docker)
- Fine granular configurability for admins in ``/etc/userdocker/`` allows to:

  - restrict runnable images if desired (allows admin reviews)
  - restrict run to locally available images
  - restrict available mount points (or enforce them, or default mount)
  - probe mounts (to make sure nfs automounts don't make docker sad)
  - enforce non-root user in container (same uid:gid as on host)
  - enforce dropping caps
  - enforce environment vars
  - enforce docker args
  - restrict port publishing
  - explicitly white-list available args to user
  - restrict allowed GPU access / reservations via ``NV_GPU``

- System wide config + overrides for individual groups, gids, users, uids.
- Easy extensibility for further subcommands and args.


Installation:
=============

The installation of userdocker works in three steps:


1. Install package:
-------------------

First make sure that docker is installed:

.. code-block:: bash

    sudo docker version

Afterwards, as userdocker is written in python3 and available as python package:

.. code-block:: bash

    sudo pip3 install userdocker

This will give you a ``userdocker`` command that you can test with:

.. code-block:: bash

    userdocker -h

The above is the preferable way of installation of the latest stable release.

If you want to try the current master (stable dev):

.. code-block:: bash

    sudo pip3 install -U https://github.com/joernhees/userdocker/archive/master.tar.gz

Alternatively (and to contribute), you can clone this repo and execute:

.. code-block:: bash

    sudo python3 setup.py install


2. Configuration:
-----------------

Copy the default config to ``/etc/userdocker/config.py``, then edit the file.
The config contains tons of comments and explanations to help you make the right
decisions for your scenario.

.. code-block:: bash

    sudo cp /etc/userdocker/default.py /etc/userdocker/config.py


3. Allowing users to run ``sudo userdocker``:
---------------------------------------------

You should now allow the users in question to run ``sudo userdocker``. This is
basically done by adding a ``/etc/sudoers.d/userdocker`` file. If you want to
grant this permission to all users in group ``users``, add the following
two lines:

::

    Defaults env_keep += "NV_GPU"
    %users ALL=(root) NOPASSWD: /usr/local/bin/userdocker

The first is strongly recommended in case you want to allow users to use nvidia
GPUs from within docker containers via nvidia-docker (see EXECUTORS in config).
Without it they cannot pass the NV_GPU environment variable to the userdocker
(and thereby nvidia-docker) command to select their desired GPU(s).


Distributed training with slurm
-------------------------------

Slurm is well-suited for distributed training. Jobs can run multiple tasks on
multiple nodes, where each task handles one or more GPUs. Userdocker can be
configured to add make setup easier.

To enable distributed training, all worker nodes need to join a docker swarm.
The head node should be swarm manager. Please refer to the docker documentation
on how to setup swarm.

Using docker swarm, userdocker allows users to create overlay networks with
the ``userdocker network create`` command. By default, users can create
networks within their own namespace, i.e., ``[username]_*``. These networks
can then be used by containers to communicate. Just add ``--network=[name]``
to your run command.

Userdocker will also set a couple of environment variables for each container,
so you can easily setup process groups:

#. The first node hosts rank0, which usually handles the organization of the
   process group. You can access rank0 with the ``USERDOCKER_RANK0_ADDRESS``
   envionment variable.
#. ``SLURM_PROCID`` gives the rank of the current process.
#. ``SLURM_NTASKS`` gives the number of tasks in the job, i.e., the world size
#. ``SLURM_NNODES`` gives the number of nodes participating in the job.
#. Slurm uses CUDA_VISIBLE_DEVICES instead of NV_GPU to assign GPUs to tasks.
   Set NV_USE_CUDA_VISIBLE_DEVICES to make userdocker use it as a fallback
   when NV_GPU is not defined.
#. Slurm does not (yet) bind GPUs to tasks. Set SLURM_BIND_GPU to let
   userdocker distribute and bind them by setting NV_GPU for the container.
#. By default, ``CUDA_VISIBLE_DEVICES`` will be setup such that GPUs are evenly
   distributed across containers.

For multi-node training with infiniband, add the devices to ADDITIONAL_ARGS,
e.g.::

    ADDITIONAL_ARGS = [
        '--device=/dev/infiniband/rdma_cm',
        '--device=/dev/infiniband/uverbs0',
        '--device=/dev/infiniband/uverbs1',
        '--device=/dev/infiniband/uverbs2',
        '--device=/dev/infiniband/uverbs3',
    ]


If nccl is used you also need to tell it which devices to use for communication
via NCCL_SOCKET_IFNAME and NCCL_IB_HCA variables::

    ENV_VARS = [
        # sets HOME env var to user's home
        'HOME=' + user_home,
        'NCCL_SOCKET_IFNAME=eth0,mlx5',
        'NCCL_IB_HCA=mlx5',
    ]


FAQ:
====

Why sudo?
---------

Because it supports logging and is in general a lot more configurable than the
alternatives. For example if you only want to make ``userdocker`` available on
some nodes in your cluster, you can use the Host\_List field:

::

    %users node1,node2,node4=(root) /usr/local/bin/userdocker

