# _*_coding : UTF-8 _*_
# @Time : 2023/6/12 11:24 
# @Author : NEFU_GaoMY
# @File : maxent混淆矩阵 
# @Project : machine learning

import geopandas as gpd
from osgeo import gdal
import numpy as np

def sample_classification_results(classification_file, validation_file):
    classification_dataset = gdal.Open(classification_file, gdal.GA_ReadOnly)
    classification_data = classification_dataset.GetRasterBand(1).ReadAsArray()

    validation_data = gpd.read_file(validation_file)
    validation_labels = validation_data["varieties"]

    sampled_classification = []
    sampled_validation = []

    geo_transform = np.array(classification_dataset.GetGeoTransform())

    for index, row in validation_data.iterrows():
        x, y = row.geometry.xy
        x = int((x - geo_transform[0]) / geo_transform[1])
        y = int((y - geo_transform[3]) / geo_transform[5])

        classification_label = classification_data[int(y), int(x)]  # 将索引值转换为整数
        validation_label = validation_labels[index]

        if classification_label >= 0 and validation_label >= 0:
            sampled_classification.append(classification_label)
            sampled_validation.append(validation_label)

    return np.array(sampled_classification), np.array(sampled_validation)

def calculate_confusion_matrix(sampled_classification, sampled_validation, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    for i in range(len(sampled_classification)):
        predicted = int(sampled_classification[i])
        actual = int(sampled_validation[i])
        confusion_matrix[actual, predicted] += 1

    return confusion_matrix

def calculate_overall_accuracy(confusion_matrix):
    diagonal_sum = np.trace(confusion_matrix)
    total_sum = np.sum(confusion_matrix)
    overall_accuracy = diagonal_sum / total_sum

    return overall_accuracy

# 示例用法
classification_file = r"E:\maxent_tif\max\maxent_fin.tif"
# validation_file = r"C:\Users\R6meg\Desktop\tree\test_add_maxent.shp"
validation_file = r"C:\Users\R6meg\Desktop\tree\GEE_test.shp"
num_classes = 5

num_classes = int(num_classes)

sampled_classification, sampled_validation = sample_classification_results(classification_file, validation_file)
confusion_matrix = calculate_confusion_matrix(sampled_classification, sampled_validation, num_classes)
# confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
overall_accuracy = calculate_overall_accuracy(confusion_matrix)
# overall_accuracy = diagonal_sum / total_sum
print("Confusion Matrix:")
print(confusion_matrix)
print("Overall Accuracy:", overall_accuracy)
