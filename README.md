# ceh-seldon-models

## Build

To build the Seldon image, first train the model using:

```shell
$ python train.py
```

And then build the image using the `s2i` with:

```shell
$ s2i build https://github.com/ruivieira/ceh-seldon-models \
  seldonio/seldon-core-s2i-python36:0.18 \
  ruivieira/ceh-seldon-models
```
