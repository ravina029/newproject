import sys
from dataclasses import dataclass 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.compose import ColumnTransformer   # used to create different pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from source.exception import CustomException
from source.logger import logging
import os

from source.utils import save_object #used for saving the pickle files

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifact","preprocessor.pkl")     #to store any model in a pickle file
    
class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        #this function is respondible for the data transformation
        try:
            numerical_column=["writing score","reading score"]  
            categorical_columns=[
                "gender", 
                "race/ethnicity", 
                "parental level of education", 
                "lunch", 
                "test preparation course"
            ] 
            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scalar",StandardScaler(with_mean=False))
                       
                       ]
            ) 
            cat_pipeline=Pipeline( steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                  ("one_hot_encoder",OneHotEncoder()),
                                  ("scalar",StandardScaler(with_mean=False))
            ]

            )
            
            logging.info("Neumerical colum standardscaling is done")
            logging.info("Categorical colum encoding is done")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading the train and test data is done")       
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math score'
            numerical_column=["writing score",'reading score']

            input_feature_train_df =train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing on training dataframe and testing dtaframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path,

            )
        
        except Exception as e:
            #logging.error(f"An error occurred: {str(e)}")
            raise CustomException(e,sys)