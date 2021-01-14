[![Build Status](https://travis-ci.com/vibinabraham/fermicluster.svg?token=xQgyGGx6x2UXYitmpAwa&branch=master)](https://travis-ci.com/vibinabraham/fermicluster)
[![codecov](https://codecov.io/gh/vibinabraham/fermicluster/branch/master/graph/badge.svg?token=OTM2X6N7GF)](https://codecov.io/gh/vibinabraham/fermicluster)

# Fermi-Cluster 
Fermi-Cluster is an open source program to run tensor product based quantum chemistry by partitioning the system into clusters. 
Fermi-Cluster uses [pyscf](http://pyscf.org/) as a back-end for molecular systems and [ray](https://ray.io/) for parallelization.

### Installation
1. Download
    
        git clone https://github.com/vibinabraham/fermicluster.git
        cd tpsci/

2. create virtual environment (optional)
         
        virtualenv -p python3 venv
        source venv/bin/activate

3. Install

        pip install .

4. run tests
    
        pytest test/*.py
