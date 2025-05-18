# Hyperbolic Hierarchical Text Classification

Code for the paper "Hierarchical Classification of Propaganda
Techniques in Slavic Texts in Hyperbolic Space" at [SlavicNLP 2025](https://bsnlp.cs.helsinki.fi/shared-task.html).

Extending XLM-RoBERTa with (hyperbolic) graph convolutions and hierarchical stretching for the node classification of
persuasion techniques in parliamentary debates and social media posts in 5 Slavic languages.

## Getting Started

Run the following commands to install all dependencies in the current environment, merge required data from
GitHub repositories, and apply a small bug fix:

```
pip3 install -r requirements.txt
git submodule init
bash merge_hie.sh
```

## Usage

The following command runs the training loop using the configurations used in the paper by augmenting the training data
with machine translations, solving the node classification task without adding an additional linear output layer,
using the full extended SemEval 2024 hierarchy, and applying hierarchical streching with λ = 1 to an HGCN with
256-dimensional node features.
```
python3 train.py --machine_translations --node_classification --hierarchy full --gnn HIE --hie_lambda 1 --node_dim 256
```
For additional parameters, refer to ```config.py```.

## Troubleshooting

If you get an error related to an undefined symbol in torch-scatter, find your installed PyTorch version
[here](https://pytorch-geometric.com/whl/index.html) and install the packages via
```
pip3 install -f https://pytorch-geometric.com/whl/torch-<version>.html
```

## Preliminary BibTeX

```
@inproceedings{ufal4demSlavicNLP2025task,
    title={Hierarchical Classification of Propaganda Techniques in Slavic Texts in Hyperbolic Space},
    author={Br{\"u}ckner, Christopher  and  Pecina, Pavel},
    booktitle = {Proceedings of the 10th Workshop on Slavic Natural Language Processing 2025 (SlavicNLP 2025)},
    editor = {Piskorski, Jakub and Ljube\v{s}i{'c}, Nikola and Marci{'n}czuk, Micha{\l} and      Nakov, Preslav and P{\v{r}}ib{'a}{\v{n}}, Pavel and Yangarber, Roman},
    month = {July},
    year = {2025},
    address = {Vienna, Austria},
    publisher = {Association for Computational Linguistics}          
}
```

## Acknowledgments

* [HGCN](https://github.com/HazyResearch/hgcn)
* [HIE](https://github.com/marlin-codes/HIE)
* [TWIN4DEM](https://twin4dem.eu)
