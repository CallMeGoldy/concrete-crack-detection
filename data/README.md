# Data

The dataset is **not** included in this repository.

This project uses 10,000 greyscale 30×30 px concrete surface images (5,000 with cracks, 5,000 without). They are a downscaled subset of the
[Concrete Crack Images for Classification](https://data.mendeley.com/datasets/5y9wdsg2zt/2) dataset by Çağlar Fırat Özgenel (Mendeley Data), and were supplied as `Concrete_Crack_Images.zip` for the course case study.

## Expected layout

Put the zip file here:

```
data/
└── Concrete_Crack_Images.zip
    └── Concrete_Crack_Images/
        ├── Negative_img/   # 00001.jpg … 05000.jpg (no crack)
        └── Positive_img/   # 00001.jpg … 05000.jpg (crack)
```

Each notebook's Step 2 extracts it automatically, either from `data/` when run locally or from an upload when run on Google Colab.

If you're starting from the original full-resolution dataset (227×227 RGB), both notebooks still work: the SVM notebook resizes images to 30×30. For the CNN, add a `resize((30, 30))` in `Concrete_Img.__getitem__` so that the layer sizes match.
