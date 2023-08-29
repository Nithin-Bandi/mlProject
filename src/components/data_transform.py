import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustopException
from src.logger import logging
from src.utils import save_object
@ dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self) :
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):

        '''
        This fun is responsible for data transformation

        '''

        try:
            numerical_cols=['writing_score','reading_score']
            catogerical_cols=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
#gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
            ]
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
                
                
                )
            catogerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False)),
                ]

            )
            logging.info(f"Categorical columns: {catogerical_cols}")
            logging.info(f"Numerical columns:{numerical_cols}")
            
            logging.info('Categorical columns encoding completed')
          
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('catogerical_pipeline',catogerical_pipeline,catogerical_cols)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustopException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading train and test data completed")
            logging.info("Obtain preprocessing Object")

            preprocessor_obj=self.get_data_transformer_obj()

            target_column_name='math_score'
            numerical_column=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing objects on training dataframe and testing dataframe")

            

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing obkect")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

            

        except Exception as e:
            raise CustopException(e,sys)

