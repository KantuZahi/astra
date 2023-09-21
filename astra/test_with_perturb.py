# -*- encoding: utf-8 -*-
import os
import platform
import sys
import argparse
import numpy as np
import SimpleITK as sitk
import skimage.morphology as skm
from tqdm import tqdm

if os.path.abspath("..") not in sys.path:
    sys.path.insert(0, os.path.abspath(".."))

from astra.utils.data_utils import (
    read_data,
    pre_processing,
    test_time_augmentation,
    copy_sitk_imageinfo,
)
from astra.model.model import Model
from astra.training.network_trainer import *


def find_boundary_points(volume):
    """
    Find points on the boundary of a region of interest.
    These points will then be used to create perturbations.
    """
    ball = skm.ball(2)
    volume_larger = skm.binary_dilation(volume[0, :, :, :], ball)
    boundary_volume = volume_larger - volume[0, :, :, :]
    points = np.nonzero(boundary_volume)
    out_points = []

    # Choose 10 here to sub-sample the surface. Need to think of a better way to do this.
    for idx in range(0, len(points[0]), 5):
        x = points[0][idx]
        y = points[1][idx]
        z = points[2][idx]
        out_points.append([x, y, z])
    return out_points

def find_boundary_points_CTV(volume):
    """
    Find points on the boundary of a region of interest.
    These points will then be used to create perturbations.
    """
    ball = skm.ball(2)
    volume_smaller = skm.binary_erosion(volume[0, :, :, :], ball)
    boundary_rim = volume[0, :, :, :] - volume_smaller
    points = np.nonzero(boundary_rim)
    out_points = []

    # Choose 10 here to sub-sample the surface. Need to think of a better way to do this.
    for idx in range(0, len(points[0]),3):
        x = points[0][idx]
        y = points[1][idx]
        z = points[2][idx]
        out_points.append([x, y, z])
    return out_points

def dilate_at(volume, point):
    """
    Dilate the binary volume 'volume' at the point specified bt point.
    """
    ball = skm.ball(3)
    point_vol = np.zeros(volume[0, :, :, :].shape, dtype=np.uint8)
    point_vol[point[0], point[1], point[2]] = 1
    volume_out = skm.binary_dilation(point_vol, ball).astype(np.uint8)
    volume_out += volume[0, :, :, :].astype(np.uint8)
    volume_out[volume_out >= 1] = 1
    volume_out = volume_out[np.newaxis, :, :, :]
    return volume_out

def erode_at(volume, point):
    """
    Erode the binary volume 'volume' at the point specified bt point.
    """
    ball = skm.ball(3)
    point_vol = np.zeros(volume[0, :, :, :].shape, dtype=np.uint8)
    point_vol[point[0], point[1], point[2]] = 1
    volume_out = skm.binary_dilation(point_vol, ball).astype(np.uint8)
    volume_out = volume[0, :, :, :].astype(np.uint8) - volume_out
    volume_out[volume_out >= 2] = 0
    volume_out = volume_out[np.newaxis, :, :, :]
    return volume_out

def inference_with_perturbation(trainer, list_patient_dirs, save_path, do_TTA=True):
    """
    This function helps create perturbations in the OAR and the Target, and then evaluates the dose.
    """
    sys = platform.system()

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in list_patient_dirs:
            if sys == 'Windows':
                patient_id = patient_dir.split("\\")[-1]
            else:
                patient_id = patient_dir.split("/")[-1]
            dict_images = read_data(patient_dir)

            list_images = pre_processing(dict_images)

            input_ = list_images[0]
            possible_dose_mask = list_images[1]

            # Test-time augmentation
            if do_TTA:
                TTA_mode = [[], ["Z"], ["W"], ["Z", "W"]]
            else:
                TTA_mode = [[]]
            prediction = test_time_augmentation(trainer, input_, TTA_mode)

            # Pose-processing
            prediction[
                np.logical_or(possible_dose_mask[0, :, :, :] < 1, prediction < 0)
            ] = 0
            gt_prediction = 70.0 * prediction

            templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
            prediction_nii = sitk.GetImageFromArray(gt_prediction)
            prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)


            if sys == 'Windows':
                if not os.path.exists(save_path + "\\" + patient_id):
                    os.mkdir(save_path + "\\" + patient_id)
                sitk.WriteImage(
                    prediction_nii,
                    save_path + "\\" + patient_id + "/Dose_gt.nii.gz",
                )
            else:
                if not os.path.exists(save_path + "/" + patient_id):
                    os.mkdir(save_path + "/" + patient_id)
                sitk.WriteImage(
                    prediction_nii,
                    save_path + "/" + patient_id + "/Dose_gt.nii.gz",
                )

            list_target = ["Target"]
            list_oar_names = ["BrainStem", "Hippocampus_L", "Hippocampus_R"] # , "Eye_L", "Eye_R", "Chiasm", "Cochlea_L", "Cochlea_R", "LacrimalGland_L", "LacrimalGland_R", "OpticNerve_L", "OpticNerve_R", "Pituitary"]


            for organ in list_target:

                print("Working on: ", organ.split("_")[0])

                perturb_prediction = {}
                perturb_prediction[organ] = np.zeros_like(gt_prediction)

                for oar in list_oar_names:
                    perturb_prediction[oar] = np.zeros_like(gt_prediction)

                prediction_tv = np.zeros_like(gt_prediction)
                prediction_tv += np.multiply(gt_prediction, dict_images[organ][0,:,:,:])

                templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
                prediction_nii = sitk.GetImageFromArray(prediction_tv)
                prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
                if sys == 'Windows':
                    if not os.path.exists(save_path + "\\" + patient_id):
                        os.mkdir(save_path + "\\" + patient_id)
                    sitk.WriteImage(
                        prediction_nii,
                        save_path + "\\" + patient_id + "/Prediction_NoPert" + organ + ".nii.gz",
                    )
                else:
                    if not os.path.exists(save_path + "/" + patient_id):
                        os.mkdir(save_path + "/" + patient_id)
                    sitk.WriteImage(
                        prediction_nii,
                        save_path + "/" + patient_id + "/Prediction_NoPert" + organ + ".nii.gz",
                    )

                ### Use this to get boundary on the CTV
                point_set = find_boundary_points_CTV(dict_images[organ])

                ### Use this to get boundary on a imaginary PTV at n voxels away from the CTV
                # point_set = find_boundary_points(dict_images[organ])

                print("\n Points on surface: ", len(point_set))


                # At this stage, do perturbation on the organ boundary.
                for point in tqdm(point_set):

                    ### put CTV into erode/ dilate function
                    dict_images[organ] = dilate_at(dict_images[organ], point)
                    # dict_images[organ] = erode_at(dict_images[organ], point)
                    test_dict = dict_images[organ]


                    list_images = pre_processing(dict_images)

                    input_ = list_images[0]
                    possible_dose_mask = list_images[1]

                    # Test-time augmentation
                    if do_TTA:
                        TTA_mode = [[], ["Z"], ["W"], ["Z", "W"]]
                    else:
                        TTA_mode = [[]]
                    prediction = test_time_augmentation(trainer, input_, TTA_mode)

                    # Pose-processing
                    prediction[
                        np.logical_or(
                            possible_dose_mask[0, :, :, :] < 1, prediction < 0
                        )
                    ] = 0
                    # rescale and get gray (Gy)
                    prediction = 70.0 * prediction

                    # max/mean value of oar written into perturb location
                    for oar in list_oar_names:
                        temp_pred_gt = np.multiply(gt_prediction, dict_images[oar])
                        temp_pred_pert = np.multiply(prediction,dict_images[oar])
                        absdiff = np.sum(abs(temp_pred_gt - temp_pred_pert))
                        # max_val_pert = np.max(temp_pred_pert)
                        # max_val_gt = np.max(temp_pred_gt)
                        max_gt = np.max(temp_pred_gt)
                        max_pert = np.max(temp_pred_pert)
                        deltamax = np.max(np.abs(temp_pred_gt - temp_pred_pert))

                        perturb_prediction[oar][point[0], point[1], point[2]] = deltamax



                    absdiff = np.sum(np.abs(np.multiply(gt_prediction, dict_images[organ]) - np.multiply(prediction,dict_images[organ])))
                    deltamax = np.max(np.abs(np.multiply(gt_prediction, dict_images[organ]) - np.multiply(prediction,dict_images[organ])))
                    # perturb_prediction[organ][point[0], point[1], point[2]] = absdiff
                    perturb_prediction[organ][point[0], point[1], point[2]] = deltamax


                for oar in list_oar_names:
                    templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
                    prediction_nii = sitk.GetImageFromArray(perturb_prediction[oar])
                    prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
                    if sys == 'Windows':
                        if not os.path.exists(save_path + "\\" + patient_id):
                            os.mkdir(save_path + "\\" + patient_id)
                        sitk.WriteImage(
                            prediction_nii,
                            save_path + "\\" + patient_id + "/Perturbed_T" + oar + ".nii.gz",
                        )
                    else:
                        if not os.path.exists(save_path + "/" + patient_id):
                            os.mkdir(save_path + "/" + patient_id)
                        sitk.WriteImage(
                            prediction_nii,
                            save_path + "/" + patient_id + "/Perturbed_T" + oar + ".nii.gz",
                        )

                templete_nii = sitk.ReadImage(patient_dir + "/Dose_Mask.nii.gz")
                prediction_nii = sitk.GetImageFromArray(perturb_prediction[organ])
                prediction_nii = copy_sitk_imageinfo(templete_nii, prediction_nii)
                if sys == 'Windows':
                    if not os.path.exists(save_path + "\\" + patient_id):
                        os.mkdir(save_path + "\\" + patient_id)
                    sitk.WriteImage(
                        prediction_nii,
                        save_path + "\\" + patient_id + "/Perturbed_" + organ + ".nii.gz",
                    )
                else:
                    if not os.path.exists(save_path + "/" + patient_id):
                        os.mkdir(save_path + "/" + patient_id)
                    sitk.WriteImage(
                        prediction_nii,
                        save_path + "/" + patient_id + "/Perturbed_" + organ + ".nii.gz",
                    )






if __name__ == "__main__":

    root_dir = "/home/studentshare/Documents/astra/"
    # root_dir = os.getcwd()
    model_dir = os.path.join(root_dir, "models")
    output_dir = os.path.join(root_dir, "output_perturb")
    os.makedirs(output_dir, exist_ok=True)

    gt_dir = os.path.join(root_dir, "data", "processed-dldp")
    test_dir = gt_dir  # change this if somewhere else.

    if not os.path.exists(model_dir):
        raise Exception(
            "OpenKBP_C3D should be prepared before testing, please run prepare_OpenKBP_C3D.py"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--GPU_id", type=int, default=-1, help="GPU id used for testing (default: 0)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(model_dir, "best_val_evaluation_index.pkl"),
    )
    parser.add_argument(
        "--TTA", type=bool, default=True, help="do test-time augmentation, default True"
    )
    args = parser.parse_args()

    trainer_ = NetworkTrainer()
    trainer_.setting.project_name = "C3D"
    trainer_.setting.output_dir = output_dir

    trainer_.setting.network = Model(
        in_ch=15,
        out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        list_ch_B=[-1, 32, 64, 128, 256, 512],
    )

    # Load model weights
    trainer_.init_trainer(
        ckpt_file=args.model_path, list_GPU_ids=[args.GPU_id], only_network=True
    )

    for subject_id in [90, 82, 81]:

        # Start inference
        print("\n\n# Start inference !")
        list_patient_dirs = [os.path.join(test_dir, "DLDP_" + str(subject_id).zfill(3))]
        inference_with_perturbation(
            trainer_,
            list_patient_dirs,
            save_path=os.path.join(trainer_.setting.output_dir, "Prediction_D3_DMax"),
            do_TTA=args.TTA,
        )
