# mev-hcl

This repository contains a WebMeV-compatible tool for performing hierarchical clustering. To be run using the Cromwell engine using the WDL-format file.

Note that we use the Cromwell engine since it allows us to dynamically provision machines capable of clustering along the gene/feature dimension (often >30k features). Without that, the clustering will overwhelm the modest memory resources of the WebMeV server.

---

### To run external of WebMeV:

Either:
- build the Docker image using the contents of the `docker/` folder (e.g. `docker build -t myuser/hcl:v1 .`) 
- pull the docker image from the GitHub container repository (see https://github.com/web-mev/mev-hcl/pkgs/container/mev-hcl)

To run, enter the container in an interactive shell:
```
docker run -it -v$PWD:/work <IMAGE>
```
(here, we mount the current directory to `/work` inside the container)

Then, run the clustering script:
```
python3 /opt/software/run_hcl.py -i <input_matrix path> \
            -d <distance_metric> \
            -l <linkage> \
            -c <clustering_dimension> \
            -o <observations_output path> \
            -f <features_output path>
```
(or `python3 /opt/software/run_hcl.py -h` for help)

The `-i` argument is the path to an input file path, which is a matrix of (genes, samples). Note that this might be different than the typical convention (in statistical learning) of having each *row* correspond to an observation and each *column* to a feature. In our genomics context, each *column* corresponds to the expression profile for a sample/observation, so we are transposed from the canonical orientation.

Explore the `run_hcl.py` script to dig deeper on the use/meaning of `linkage` and `distance_metric`.

The outputs of the process are JSON-format files which describe the hierarchical tree structure of the clustering. If only one of the dimensions is chosen, the other file will contain an empty tree `{}`.