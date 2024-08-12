# PaScal Analyzer
A tool and library from the Parallel Scalability Suite toolset to control, measure, and collect information from multiple executions of a parallel application. Currently suited only for shared-memory environments.

## Table of contents

- [What is PaScal Analyzer](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Options available](#options-available)



## Description

**PaScal Analyzer** simplifies the execution and measurement of several executions of a parallel program, allowing the analysis of scalability trends in configuration environments with different amounts of processing elements and different workloads. Its output can be opened by the PaScal Viewer tool to assist the user in the understanding of the program’s behavior with visual elements that emphasize scalability bottlenecks that may require more in-depth analyzes. At the current version, the PaScal Analyzer can be used to help develop programs that run on a single shared-memory computational node. Future versions will support distributed programs running on many nodes.
  
The tool has a low level of intrusion in the programs under analysis, which is a fundamental aspect in understanding their behavior and scalability capacity accurately.

### Features
  - Runs applications with multiple different frequencies, number of cores, and inputs with a given number of repetitions;
  - Calculate the speedups and efficency of applications, if it's possible, using the measured times of execution.

### Prerequisites
  - libpfm4
  - g++
  - swig
  - make
  - Python3 or newer
    - cpufreq
    - netipmi
    - psutil
    - numpy
    - pandas
    - scipy
    - sklearn
    - ghalton
    - performance-features

## Installation

Make sure to use the steps below to download and install the tool. If you use git to clone, make sure to use LFS option to include large binary files.

```bash
wget -c https://gitlab.com/lappsufrn/pascal-releases/-/archive/master/pascal-releases-master.zip
unzip pascal-releases-master.zip
rm pascal-releases-master.zip
cd pascal-releases-master/
./install.sh
```

## Usage

Running the tool from the command line:

  ```bash
  pascalanalyzer ./myapp --inst aut --idtm 5 --cores 1,4  --frqs 3000000,2800000 --verb 3
  pascalanalyzer ./myapp -c 1,32 -v 2 -o myoutput.json
  ```

### Data analysis library

  Class used to generate the measured times structure, to save such data in a
  "json" file, to load a previously saved json data file, to calculate the
  speedup or efficiency of application and to plot 2D or 3D graph of time,
  speedup or efficiency versus the number of cores and frequency or input size.
  ```python
  from pascalanalyzer import PascalData
  d = PascalData("path_to_datafile")
  print(d)        # Print summary informations
  d.times()       # Show a Dataframe with mesures times
  d.speedups()    # Show a Dataframe with speedups
  d.energy()      # Show a Dataframe with energy
  ```

### Instrumenting source code

Manual instrumentation is avialable using the functions `pascal_start(id)` and `pascal_stop(id)` defined at `include/pascalops.h`.

- `pascal_start`: receive 1 arguments, the identification of the region.
It marks the beginning of the parallel area to be measured.
- `pascal_stop`: receive 1 arguments, the identification of the region.
It marks the end of the parallel area to be measured.

Example:

```C++
#include "pascalops.h"
...
int main()
{
    ...
	pascal_start(1);
	#pragma omp parallel
	{
		...
	}
	pascal_stop(1);
	...
}
```


A option to automatically instrument OpenMP code is also available using the command line:
`./pascalanalyzer --imnt teste/ .cpp,.c`, this will mark all `pragma omp parallel` regions on the cpp and c files on the folder teste.

## Options available

  Command line tool to run application collecting information while executing
  program with different configurations of cores, frequencies, inputs arguments.

    usage: pascalanalyzer [-h] [-c CORES] [-f FREQUENCIES] [-i INPUTS] [-o OUTPUT] [-r RPTS] [-g]
                      [-t {aut,AUT,man,MAN}] [-a LEVEL] [--mpi RUNTIME] [--ragt TYPE] [-v VERBOSE]
                      [--dcrs] [--dhpt] [--domp] [--dout] [--govr GOVERNOR] [--idtm TIME]
                      [--fgpe EVENT] [--fgps EVENT] [--rple {sysfs,perf}]
                      [--rpls {sysfs,scontrol,perf}] [--ipmi SERVER USER PASSWORD] [--modl NPTS MODE]
                      [--prcs] [--lpcs] [--imnt PATH EXTENSIONS]
                      [application]

    Script to run application collecting information while executing a program with different
    configurations of cores, frequencies, and inputs arguments.

    positional arguments:
      application           Application name to run

    optional arguments:
      -h, --help            show this help message and exit
      -c CORES, --cors CORES
                            List of cores numbers to be used. Ex: 1,2,4. For scalability analysis, organize the number of cores 
                            from the smallest to the largest and make sure the ratio between adjacent core numbers is consistent 
                            with the ratios for the -i argument.
      -f FREQUENCIES, --frqs FREQUENCIES
                            List of frequencies (KHz). Ex: 2000000,2100000
      -i INPUTS, --ipts INPUTS
                            Input arguments to be used separated by commas. For scalability analysis, organize the input arguments 
                            from the smallest problem size to the largest and sure the ratio between adjacent problem sizes is 
                            consistent with the ratios for the -c argument.
      -o OUTPUT, --outp OUTPUT
                            Output file name
      -r RPTS, --rpts RPTS  Number of repetitions for a specific run. (Default: 1)
      -g, --gapr            Include as regions the gaps among regions already identified
      -t {aut,AUT,man,MAN}, --inst {aut,AUT,man,MAN}
                            Instrumentation type to identify code regions (auto or manual). When not used, the tool will not 
                            attempt to identify inner regions.
      -a LEVEL, --ragl LEVEL
                            Set the regions hierarchy level to aggregate measures. 0 = No level.
      --mpi RUNTIME         Enable mpi and set the runtimer
      --ragt TYPE           Region aggregation type. The 'acc' type means to use a accumulated time
                            variable that reduces memory consumption and calculation time.
      -v VERBOSE, --verb VERBOSE
                            verbosity level. 0 = No verbose
      --dcrs                Disable cores
      --dhpt                Enable hyperthread (disabled by default)
      --domp                Set OMP_NUM_THREADS variable with -c parameter values. (enabled by default)
      --dout                Discard the output
      --govr GOVERNOR       Set the cpu governor
      --idtm TIME           Idle time between runs. (Default: 0)
      --fgpe EVENT          Collect performance counters
      --fgps EVENT          Sample performance counters
      --rple {sysfs,perf}   Enable rapl energy measuments
      --rpls {sysfs,scontrol,perf}
                            Enable rapl sample energy measuments
      --ipmi SERVER USER PASSWORD
                            Enable ipmi measuments
      --modl NPTS MODE      Run pascal with random configurations to create a model
      --prcs                Track cpu cores
      --lpcs                List performance counters
      --imnt PATH EXTENSIONS
                            Instrument code with pascal_start() and pascal_stop() around OpenMP parallel regions



## Site

- <https://lappsufrn.gitlab.io/pascalsuite-releases/>

### License

PaScal-releases © 2022 is licensed under Attribution-NonCommercial-NoDerivatives 4.0 International 


