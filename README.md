# image-segmentation

## Set up enviroments
1. Fill out the library in enviroment.yml file
2. Create condata enviroments with prefix
``` conda env create --prefix ./env --file environment.yml # create the environment ```
3. Activate enviroments
``` conda activate ./env # activate the environment ```
4. Check package list
``` conda list```
5. Check version of the package
``` conda list | grep numpy```
6. Update package
``` conda env update --prefix ./env --file environment.yml --prune) # update the environment ```
7. [For more information about conda](https://kaust-vislab.github.io/python-novice-gapminder/00-getting-started-with-conda/index.html)


## Other check CUDA
1. cuda version: ``` nvidia-smi ```
2. cudnn version: ``` /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2 ```
3. cudatoolkit version: ``` nvcc --version ```

