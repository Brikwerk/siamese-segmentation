# Siamese Neural Network for Object  Co-Segmentation

This project aims to generate predictive background/foreground masks from image pairs with similar objects. [CoSegNet's architecture](https://www.ijcai.org/proceedings/2019/0095.pdf) was used as the basis for this project. 

## Getting Started

First, create the "icoseg_data" directory at this project's root and then download/extract the [iCoseg dataset](http://chenlab.ece.cornell.edu/projects/touch-coseg/) (what I used to train, feel free to use another alternative dataset) into the folder.

Next, install all the python requirements:

```bash
pip install -r requirements.txt
```

To enable the tqdm progress bar in the Jupyter Notebook, please run the following command after dependencies are installed:

```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

If you wish to train the model and evaluate it, run:

```bash
jupyter notebook
```
and open up the "train.ipynb" file.

The model, datasets, and other utilities can be found in the src folder.