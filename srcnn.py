import rasterio
from rasterio.transform import from_origin
import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint
# from tifffile import tiffinfo
from model import get_model
from preprocess import preprocess_dataset
from util import clean_mkdir, load_data


def train(data_path, model_path, epochs=32, batch_size=16):
    # preprocess_dataset(data_path)
    train_path = str(data_path + "/train")
    train_labels_path = str(data_path + "/train_labels")
    clean_mkdir("checkpoints")
    checkpointer = ModelCheckpoint(
        filepath="checkpoints/satil2.h5", verbose=1, save_best_only=True
    )
    
    model = get_model()

    x, y = load_data(train_path, train_labels_path)
    print(x, y, "HIII")
    model.fit(
        y,
        x,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        shuffle=True,
        callbacks=[checkpointer],
    )
    model.save(model_path)


def test(data_path, model_weights_path):
    test_path = str(data_path + "/test")
    test_labels_path = str(data_path + "/test_labels")
    model = get_model(model_weights_path)
    x, y = load_data(test_path, test_labels_path)
    score = model.evaluate(x, y)
    print(model.metrics_names, score)


def get_files(data_path):
    inp_arr = []
    for file in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, file)):
            inp_arr.append(file)
    return inp_arr


def load_image(img_path):
    x = []
    img = Image.open(img_path)
    img_array = np.asarray(img, dtype="uint8")
    img_array = img_array / (255 * 1.0)
    x.append(img_array)
    return np.array(x)


# def copy_geotiff_metadata(input_img_path, output_img_path):
#     with rasterio.open(input_img_path) as src:
#         # Create a new profile for the output image with the same geospatial information
#         profile = src.profile.copy()

#     # Set the data type for the new image to store the geographic data
#         profile['dtype'] = 'float64'  # You can adjust the dtype as needed

#     # Create an array filled with geographic information
#         geo_data = src.read(1)  # You can use any band or change this as needed

#     # Open the output image for writing
#         with rasterio.open(output_img_path, 'w', **profile) as dst:
#             # Write the geographic data to the output image
#             dst.write(geo_data, 1)


def run(data_path, model_weights_path, output_path):
    output_path = Path(output_path)
    model = get_model(model_weights_path)
    inp_arr = get_files(data_path)
    for index, img_name in enumerate(inp_arr):
        x = load_image(os.path.join(data_path, img_name))

        out_array = model.predict(x)
        num, rows, cols, channels = out_array.shape
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    if out_array[0][i][j][k] > 1.0:
                        out_array[0][i][j][k] = 1.0

        out_img = Image.fromarray(np.uint8(out_array[0] * 255))
        out_img.save(str(output_path / "{}.jpg".format(img_name[:-4])))
        # copy_geotiff_metadata(os.path.join(data_path, img_name), str(
        #     output_path / "{}.tif".format(img_name[:-4])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate/run SRCNN models")
    parser.add_argument(
        "--action",
        type=str,
        default="test",
        help="Train or test the model.",
        choices={"train", "test", "run"},
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Filepath of a saved model to use for eval or inference or"
        + "filepath where to save a newly trained model.",
    )
    parser.add_argument(
        "--output_path", type=str, help="Filepath to output results from run action"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--data_path",
        type=str,
        help="Filepath to data directory. Image data should exist at <data_path>/images",
        default="data",
    )
    params = parser.parse_args()
    if params.action == "train":
        train(params.data_path, params.epochs,
              params.batch_size, params.model_path)
    elif params.action == "test":
        test(params.data_path, params.model_path)
    elif params.action == "run":
        run(params.data_path, params.model_path, params.output_path)
