# PEFT4vision

### Developed for use in CSCI 444 and to serve as a resource/tutorial for other students.

# Features

- **Split dataset into testing and training splits.**
- **Parameter-efficiently fine tune Google SigLip on your image dataset.**
- **Extract image embeddings using the PEFT-adapted model.**


# Installation
Clone this repository:
```bash
git clone https://github.com/johnkxl/peft-vision.git
```
Create a virtual environment using:
```bash
bash install_local.sh <df_analyze_path>
```
where `df_analyze_path` is the path to already cloned [df-analyze](https://github.com/stfxecutables/df-analyze) repo. This will install df-analyze as a package to use its embedding extraction wth the PEFT adapted model.
```bash
source .peft_venv/bin/activate 
```
**NOTE**: Package installing doesn't always work when running `install_local.sh`, so just ignore complaints output in the terminal and use `pip install package_name` for each package missing or being complained about.

# Usage
## Clean Dataset

Remove undersampled target variables with the following command
```bash
python drop_samples.py \
    --df <dataset_path> \
    --target <target> \
    --out <outname>
```
This ensures `df-analyze` has sufficient samples for all target classes.


## Split Dataset

Split your dataset into training and testing sets. The training set is recommended to be 90% of your dataset, with the remaining 10% for testing. Your dataset should be initial in stored in a `.parquet` file with an `images` column of type `bytes`. To split your dataset, run the following command:
```bash
python split_ds.py \
    --df DF \
    --target <TARGET> \
    --train_size <train_size> \
    --groupby [grouper] \
    --outdir <OUTDIR>
```
The recommended `train_size` is `0.9`.

The dataset splits will be stored in a directory indicated by `OUTDIR` with the following structure:
```plaintext
📂 OUTDIR/
├── label2id.json
├── train.parquet
├── train_image_target.parquet
├── test.paqrquet
└── test_image_target.parquet
```
The files with the `image_target` suffix contain only the `image` column and whicever target column was specified, renamed `target`.


## Download Model

To download the
[SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip)
model, simply run the command
```bash
python download_model.py
```
The model and image preprocessor will be stored in the directory with the following structure:
```plaintext
📂 ./downloaded_models/siglip_so400m_patch14_384
├── 📂 model/
│   ├── config.json
│   └── model.safetensors
└── 📂 preprocessor/
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── spiece.model
    └── tokenizer_config.json
```
## Train PEFT Adapter
To train the PEFT adapter, use the following command:
```bash
python train_peft.py \ 
    --train_ds <TRAIN_DS> \
    --test_size [test_size=0.111] \
    --num_epochs [num_epochs=5] \
    --learn_rate [learn_rate=5e-5] \
    --batch_size [batch-size=16] \
    --log_interval [log_interval=10] 
```
For more information on the CLIs, use `python train-peft.py --help`

The adapter is saved in the `./downloaded_models/siglip_so400m_patch14_384` directory as
```plaintext
📂 ./downloaded_models/siglip_so400m_patch14_384
├── 📂 peft_model/
│   ├── config.json
│   └── model.safetensors
└── ...
```

## Extract Embeddings
To extract image embeddings from your dataset use 
```bash
python peft-embed.py \
    --df <df_path> \
    --out <outname.parquet>
```
Make sure `df` is the dataset dedicated to testing. Using the file from the earlier split, it should be called `test_image_target.parquet`.

# Support

If you have issues running the software, contact [x2022awi@stfx.ca](mailto:x2022awi@stfx.ca).

# Citation
```bibtex
@software{jkendall-peft4vision,
  author = {John Kendall},
  title = {PEFT4vision},
  year = {2024},
  url = {https://github.com/johnkxl/peft4vision},
  version = {1.0.0}
}
```

# License

This project is licensed under the MIT license.