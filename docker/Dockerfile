ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"
# ARG MMCV="1.5.0"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libsparsehash-dev sudo wget ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN cd /workspace/; conda install  torchvision torchaudio cudatoolkit -c pytorch;\
  conda install pip opencv pillow=6.2.0 scipy scikit-image -y;\
  pip install chamferdist open3d torchpack pyflann-py3;\
  git clone https://github.com/NVIDIA/apex;cd apex;pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./;\
  pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

WORKDIR /workspace/semalign/
COPY . /workspace/semalign/



# conda install pip
# pip install chamferdist
# conda install opencv -y
# conda install pillow=6.2.1  scipy scikit-image
# pip install open3d
# sudo apt-get install libsparsehash-dev
# pip install torchpack
# pip install pyflann-py3

# pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
# pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# conda install  torchvision torchaudio cudatoolkit -c pytorch

# pip install scikit-image


# RUN conda install -y numba opencv;\
#   pip install -y torchpack ;\
#   pip install -y --upgrade git+https://github.com/mit-han-lab/torchsparse.git;\
#   git clone https://github.com/mit-han-lab/spvnas.git

# RUN git clone -b sdcnet https://github.com/NVIDIA/semantic-segmentation.git;\
#   conda install sklearn pillow h5py piexif tqdm tensorboardX;

# apt-get install libsparsehash-dev

# conda install  torchvision torchaudio -c pytorch
# conda install numba opencv
# conda install mpi4py, opencv
# pip install opencv-python
# pip install apex tensorboardX natsort
# RUN pip install --no-cache-dir --upgrade pip wheel setuptools
# RUN pip install open3d;pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html;\
#     git clone https://github.com/open-mmlab/mmdetection.git;\
#     cd mmdetection;\
#     git checkout v2.24.0  # switch to v2.24.0 branch;\
#     pip install -r requirements/build.txt;\
#     pip install -v -e . ;\
#     cd ..;\
#     git clone https://github.com/open-mmlab/mmsegmentation.git;\
#     cd mmsegmentation;\
#     git checkout v0.20.0  # switch to v0.20.0 branch;\
#     pip install -e . ;\
#     cd ..;\
#     # Install MMDetection3D
#     # RUN conda clean --all
#     COPY . /mmdetection3d
# WORKDIR /mmdetection3d
# RUN pip install openmim;mim install mmdet;mim install mmsegmentation;pip install --ignore-installed PyYAML;\
#     pip install -r requirements/build.txt;\
#     pip install --no-cache-dir -e .;\
#     pip install cumm-cu114; \
#     pip install spconv-cu114;\
