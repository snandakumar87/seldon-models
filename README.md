# ceh-seldon-models

## Build

Optionally, you can first train the model using:

```shell
$ pip install -r requirements.txt
$ python train.py
```

This repository contains a pre-trained model, so that you can build the image using the `s2i` as:

```shell
$ s2i build https://github.com/ruivieira/ceh-seldon-models \
  seldonio/seldon-core-s2i-python36:0.18 \
  ruivieira/ceh-seldon-models
```

## Deploy

Deploy on OpenShift with:

```shell
oc new-app ruivieira/ceh-seldon-models
```