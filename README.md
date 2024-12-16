# Learn How to Query from Unlabeled Data Streams in Federated Learning


This is the implementation of our paper "[Learn How to Query from Unlabeled Data Streams in Federated Learning](https://arxiv.org/abs/2412.08138)" in AAAI 2025.


## Installation instructions

Install Python environment with conda:

```bash
conda create -n leadq
conda activate leadq
```

Then install the Python packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```
NOTE: you may need to check the version of some packages such as torch.

## Run an experiment 

```shell
python main.py --al_method=[strategy name] --model=[model name] --dataset=[dataset name] --gpu=[gpu id] \
    --n_arrive=[number of arrived unlabeled samples] --n_query=[number of queried samples] 
```

| Argument       | Description   | Choices                              |
|----------------|---------------|--------------------------------------|
| `al_method`   | The data querying strategy   | leadq, kafal, logo, coreset  |
| `model`    | The name of the model  |  cnnvconv, resnet18 |
| `dataset` | The name of the dataset  | svhn, cifar100 |
| `gpu`   | The id of GPU  | 0,1, ...  |
| `n_arrive`   | The number of arrived samples per client per round      | 10, 20, ... (Integer)      |
| `n_query`| The number of queried samples per client per round    | 1, 2, ... (Integer)      |


For example, run the proposed method:

```shell
python main.py --al_method=leadq --model=cnn4conv --dataset=svhn --gpu=0 --n_arrive=10 --n_query=1
```

## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@article{sun2025learn,
  title={Learn How to Query from Unlabeled Data Streams in Federated Learning},
  author={Sun, Yuchang and Li, Xinran and Lin, Tao and Zhang, Jun},
  booktitle={accepted by The 39th Annual AAAI Conference on Artificial Intelligence (AAAI)},
  year={2025}
}
```

## Contact

If you have any questions, please feel free to contact us via hiyuchang@outlook.com.

## Acknowledgements
The initial implement of this repo is based on [LoGo](https://github.com/raymin0223/LoGo). We thank the authors for their contribution.
