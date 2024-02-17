import radiomics
import numpy as np
import pandas as pd
import cv2
from radiomics import featureextractor
from DatasetForRadiomics import DatasetForRadiomics

class Radiomics(DatasetForRadiomics):
    def __init__(self, folder_original_path, folder_mask_path, csv_file, params):
        super().__init__(folder_original_path, folder_mask_path, csv_file, params)
        
    def image_radiomics(self):
        results = []
        extractor = featureextractor.RadiomicsFeatureExtractor(self.params)
        original_paths, mask_paths = self.read_paths()

        dataset_length = len(original_paths)

        for i in range(dataset_length):
            original, mask = self.path_to_image(original_paths[i], mask_paths[i])
            result = extractor.execute(original, mask)
            results.append(result)

        feature_names = self.get_featur_names(results[0])
            
        return results, feature_names
    
    def color_extraction(self, img):
        b, g, r = cv2.split(img)
        mean_b, std_b = cv2.meanStdDev(b)
        mean_g, std_g = cv2.meanStdDev(g)
        mean_r, std_r = cv2.meanStdDev(r)
        return {"mean_b": mean_b, "std_b": std_b, "mean_g": mean_g, "std_g": std_g, "mean_r": mean_r, "std_r": std_r}
    
    def image_non_radiomics(self):
        results = []
        original_paths, mask_paths = self.read_paths()
        dataset_length = len(original_paths)

        for i in range(dataset_length):
            original, mask = self.path_to_array(original_paths[i], mask_paths[i])
            # Expand dimensions of mask array
            y = np.expand_dims(mask, axis=2)
            newmask = np.concatenate((y, y, y), axis=2)
            # Multiply new mask with image
            cob = original * newmask
            result = self.color_extraction(cob)
            results.append(result)

        feature_names = self.get_featur_names(results[0])
        return results, feature_names

        
    def get_featur_names(self, dict):

        #feature_names = list(sorted(filter ( lambda k: k.startswith("original"), dict)))
        feature_names = list(sorted(filter ( lambda k: k.startswith(""), dict)))

        return feature_names
    
    def get_samples(self):
        results, feature_names = self.image_non_radiomics()
        samples = np.zeros((self.get_length(),len(feature_names)))
        for case_id in range(0 ,self.get_length()):
            a = np.array([])
            for feature_name in feature_names:
                a = np.append(a, results[case_id][feature_name])
            samples[case_id,:] = a
            
        # May have NaNs
        samples = np.nan_to_num(samples)

        return samples, feature_names
    
    def get_radiomics_df(self):
        names = self.get_names()

        df_1 = pd.DataFrame()
        df_1['names'] = pd.Series(names)  

        samples, columns = self.get_samples()

        df_2 = pd.DataFrame(data=samples, columns=columns)

        df_3 = pd.concat([df_1, df_2], axis=1)

        return df_3
        