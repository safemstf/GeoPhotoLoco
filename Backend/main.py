import Training_module
from Evaluation import Evaluate_Model


def GeoPhotoLoco():
    true_countries, pred_countries, true_regions, pred_regions = Training_module.TrainGeoPhotoLoco(resume=True)
    Evaluate_Model(true_countries, pred_countries, true_regions, pred_regions)


if __name__ == "__main__":
    GeoPhotoLoco()
