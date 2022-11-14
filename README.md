# LengthControlledGeneration
Length controlled generation of Citation Spans


The Trained Model can be downloaded from [this link](https://drive.google.com/drive/folders/1BAR3kzBXKQqNXgca3cHzUc1DmKWH_TWH?usp=share_link)

* Download and unzip the model

* Run the following command to run the length controlled generation model

```
python generate.py model_path sample_input.json
```

`model_path` refers to the path to downloaded model


Refer to the input format in `generate.py` for using your own set of research papers. You can alternatively modify the `sample_input.json` file 