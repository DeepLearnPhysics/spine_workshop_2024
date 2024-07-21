# SPINE Workshop 2024

This repository contains all necessary resources to participate in the 2024 SPINE workshop organized for the SBN and 2x2 ML reconstruction groups. This workshop aims to train new comers to use [SPINE](https://github.com/DeepLearnPhysics/spine), our machine-learning-based particle imaging detector reconstruction chain. You can find the workshop agenda [here](https://indico.slac.stanford.edu/event/8926/).

## Software environment

For the workshop, we will use [this "Docker container"](https://hub.docker.com/layers/deeplearnphysics/larcv2/ub20.04-cuda11.6-pytorch1.13-larndsim/images/sha256-afe799e39e2000949f3f247ab73fe70039fb411cb301cb3c78678b68c22e37fb?context=explore).

Some notes below:

* The image is fairly large (multiple GBs). Please download in advance if you are using it locally. It is used in both NVIDIA GPU and CPU running mode of our software.
* Supported GPUs include those with NVIDIA Volta (e.g. V100), Turing (e.g. RTX 2080Ti), and Ampere architectures (e.g. A100, RTX 3080). If you want an older architectures to be supported, such as Pascal, please [contact Kazu](mailto:kterao@slac.stanford.edu).
* We assume basic knowledge about _software container_, in particular `Docker`. If you are learning for the first time, we recommend to use/learn about `Singularity` ([website](https://singularity.hpcng.org/)) instead of `Docker`.
    * You can pull a singularity image as follows
```shell
$ singularity pull docker://deeplearnphysics/larcv2:ub20.04-cuda11.6-pytorch1.13-larndsim
```

You can now launch a shell inside the singularity with
```shell
$ singularity exec --bind /path/to/workshop/folder/ larcv2_ub20.04-cuda11.6-pytorch1.13-larndsim.sif bash
```

### Docker alternative

You can also pull the docker image using docker (easier on Mac and Windows) directly with:
```shell
$ docker pull deeplearnphysics/larcv2:ub20.04-cuda11.6-pytorch1.13-larndsim
```
To see which images are present on your system, you can use docker images. It will look something like this:
```shell
$ docker images
REPOSITORY                TAG                                     IMAGE ID       CREATED        SIZE
deeplearnphysics/larcv2   ub20.04-cuda11.6-pytorch1.13-larndsim   cd28cb3cd04b   2 months ago   20.8GB
```
to run a shell in your image, simply do:
```shell
$ docker run -i -t cd28cb3cd04b bash
```

* [Ask Francois](mailto:drielsma@slac.stanford.edu) for questions or a request for a separate tutorial if interested.

## Resources

1. The *configuration files* are packages with this repository.

2. You can find *data files* for the examples used in this workshop under:
- S3DF
```shell
/sdf/data/neutrino/public_html/spine_workshop/larcv/ # Example MPV/MPR LArCV files prior to reconstruction
/sdf/data/neutrino/public_html/spine_workshop/reco/  # Reconstructed HDF5 files
```
- Public
  - [Small LArCV files](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/larcv/) (Day 1)
    - [Generic](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/larcv/generic_small.root)
    - [ICARUS](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/larcv/icarus_small.root)
    - [SBND](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/larcv/sbnd_small.root)
    - [2x2](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/larcv/2x2_small.root)
  - [Small corresponding reconstructed HDF5 files](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/reco/) (Day 2)
    - [Generic](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/reco/generic_small_spine.h5)
    - [ICARUS](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/reco/icarus_small_spine.h5)
    - [SBND](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/reco/sbnd_small_spine.h5)
    - [2x2](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/reco/2x2_small_spine.h5)
  - [BNB numu + cosmics](https://drive.google.com/file/d/13zSSXzWO1rsigWirtcp2vjU3EWFV4CAy/view?usp=sharing) (Day 4, 5)
  - [BNB intime cosmics](https://drive.google.com/file/d/1qBDUmCPjSsNi_SW6L6tWduPSFcBQaTMW/view?usp=sharing) (Day 4)
  - [BNB nue + cosmics](https://drive.google.com/file/d/1TwEgVMGXB8ZbrW2tdBcWFrIx4A0YTcj8/view?usp=drive_link) (Day 4)
  - [MPVMPR ee pair HDF5 file](https://drive.google.com/file/d/13x0seDs9ekQ6mwcnxUGkWJpwsis9DVRL/view?usp=sharing) (Day 5)
  - [High statistics CSV files](https://drive.google.com/drive/folders/1inRAzgCXSHEW-WAE1M25UTot_j7qioaO?usp=sharing)

3. The *network model parameters* for the inference tutorial can be found at:
- S3DF
```shell
/sdf/data/neutrino/public_html/spine_workshop/weights/generic_snapshot-4999.ckpt # Generic
/sdf/data/neutrino/public_html/spine_workshop/weights/icarus_snapshot-7999.ckpt # ICARUS
/sdf/data/neutrino/public_html/spine_workshop/weights/sbnd_snapshot-1999.ckpt # SBND
/sdf/data/neutrino/public_html/spine_workshop/weights/2x2_snapshot-3999.ckpt # 2x2
```
- Public
  - Generic: [generic_snapshot-2999.ckpt](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/weights/generic_snapshot-4999.ckpt)
  - ICARUS: [icarus_snapshot-7999.ckpt](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/weights/icarus_snapshot-7999.ckpt)
  - SBND: [sbnd_snapshot-1999.ckpt](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/weights/sbnd_snapshot-1999.ckpt)
  - 2x2: [2x2_snapshot-3999.ckpt](https://s3df.slac.stanford.edu/data/neutrino/spine_workshop/weights/2x2_snapshot-3999.ckpt)

## Computing resource
Most of the notebooks can be run strictly on CPU. The following notebooks will run significantly slower on CPU, however:
- Training/validation notebook
- Inference and HDF5 file making notebook

For all other notebooks, you can run them locally, provided that you download:
- Singularity container
- Necessary data
- [SPINE v0.1.0](https://github.com/DeepLearnPhysics/spine)

To gain access to GPUs:
- Everyone participating in this workshop should have access to both S3DF or NERSC, if you do not, please reach out to [Francois](mailto:drielsma@slac.stanford.edu).
  - SDF Jupyter ondemand: https://sdf.slac.stanford.edu/public/doc/#/
  - S3DF Jupyter ondemand: https://s3df.slac.stanford.edu/public/doc/#/

* SBN collaborators also have access to the Wilson Cluster at FNAL, equipped with GPUs. Below is a few commands to log-in and load `Singularity` with which you can run a container image for the workshop (see the previous section). For how-to utilize the Wilson Cluster, refer to [their website](https://computing.fnal.gov/wilsoncluster/slurm-job-scheduler/) as well as [this](https://cdcvs.fnal.gov/redmine/projects/nova_reconstruction/wiki/The_Wilson_Cluster) and [that](https://cdcvs.fnal.gov/redmine/projects/nova_reconstruction/wiki/Step-by-step_guide_to_running_on_the_WC) documentation from NOvA (replace `nova` with `icarus` or `sbnd` and most commands should just work).

```shell
$ ssh $USER@wc.fnal.gov
$ module load singularity
$ singularity --version
singularity version 3.6.4
```
