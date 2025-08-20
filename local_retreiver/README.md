## Local Retreiver Deployment

We refer to https://github.com/PeterGriffinJin/Search-R1, which uses wiki as retrieval corpus and build a local retriever.

### Retriever Environment Install
```python
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

### Launch Local Retrieval Server
Launch a local retrieval server for external retrieval.
```python
conda activate retriever
bash retrieval_launch.sh
```

