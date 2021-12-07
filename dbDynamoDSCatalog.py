# -*- coding: utf-8 -*-
import pandas as pd, numpy as np

import cx_Oracle
import boto3

import warnings
warnings.filterwarnings("ignore")

import re

import time
import datetime
import calendar

from pprint import pprint
from decimal import Decimal
import json
from pprint import pprint


np.set_printoptions(edgeitems=20, linewidth=5000)
pd.set_option("expand_frame_repr", True)
pd.set_option("max_colwidth", 240)


class dbDynamoDSCatalog:

    def __init__(self, dyna_conn_type='dev', orcl_conn_type='dev'):
        """
        parameters:
            dynamo_connect: ['dev' | 'prd']
                used to connect to a test table of prd table in dynamo
            conn_orcl: ['dev' | 'prd']
                used to get credentials to Oracle 'dev' or 'prd'
        """
        self.__dyn_connection_name__ = 'AWSDynamoDBConnection'

        if dyna_conn_type not in ['dev', 'prd']:
            self.__dyna_conn_type__ = 'dev'
        else:
            self.__dyna_conn_type__ = dyna_conn_type
        self.set_dynamo_table_name(self.__dyna_conn_type__)

        if orcl_conn_type not in ['dev', 'prd']:
            self.__orcl_conn_type__ = 'dev'
        else:
            self.__orcl_conn_type__ = orcl_conn_type
        self.__user__, self.__password__, self.__dsn__ = self.__get_connection_string__(self.__orcl_conn_type__)
        return


    def get_dynamo_connection_name(self):
        return self.__dyn_connection_name__


    def set_dynamo_table_name(self, dynamo_connect_type=None):
        # set dynamo table name from 'dev' or 'prd'
        table_names = {
            'dev': "",
            'prd': ""
        }
        if dynamo_connect_type not in ['dev', 'prd']:
            self.__dyna_table_name__ = table_names['dev']
        else:
            self.__dyna_table_name__ = table_names[dynamo_connect_type]
        return


    def get_dynamo_table_name(self):
        return self.__dyna_table_name__


    def __get_connection_string__(self, connect_type='dev'):
        # get connection credentials from 'dev' or 'prd'
        conn = {
            'dev': [
               "", "", ""
            ],
            'prd': [
                "", "", ""
            ]
        }

        if connect_type not in ['dev', 'prd']:
            return conn['dev']

        return conn[connect_type]


    def get_oracle_credentials(self):
        return self.__user__, self.__password__, self.__dsn__


    __model_report_dict__ = {
        'MODEL_UID': '', 'MODEL_NAME': '', 'MODEL_CREATION_DATE': '',
        'MODEL_SOURCE_ORGANIZATION': '', 'MODEL_AUTHOR': '', 'MODEL_CLASS': '',
        'MODEL_ALGORITHM_USED': '', 'MODEL_OUTPUT_UNIT': '',
        'RMSE_PCT_TRAIN_PIXELS': 0.0, 'MAE_PCT_TRAIN_PIXELS': 0.0, 'ADJUSTED_R2_TRAIN_PIXELS': 0.0,
        'RMSE_PCT_TEST_PIXELS': 0.0, 'MAE_PCT_TEST_PIXELS': 0.0, 'ADJUSTED_R2_TEST_PIXELS': 0.0,
        'RMSE_PCT_HOLDOUT_PIXELS': 0.0, 'MAE_PCT_HOLDOUT_PIXELS': 0.0, 'ADJUSTED_R2_HOLDOUT_PIXELS': 0.0,
        'TRUE_POSITIVE_RATE': 0.0, 'TRUE_NEGATIVE_RATE': 0.0,
        'FALSE_POSITIVE_RATE': 0.0, 'FALSE_NEGATIVE_RATE': 0.0,
        'AREA_UNDER_ROC_CURVE': 0.0,
        'SILHOUETTE_COEFF': 0.0, 'CALINSKI_HARABASZ_IDX': 0.0, 'NEWMAN_GIRVAN_MOD': 0.0,
        'PIXEL_COUNT_NON_LABEL_NEG_YTF': 0.0, 'BASIN_COUNT_NON_LABEL_NEG_YTF': 0.0,
        'SUM_NON_LABEL_NEG_YTF': 0.0, 'PIXEL_GLOBAL_VOL_YTF_HC': 0.0, 'BASIN_GLOBAL_VOL_YTF_HC': 0.0,
        'COMMENTS': '',
        'GR_SUFFICIENCY_SCORE': 0.0, 'GR_DENSITY_SCORE': 0.0, 'GR_DIVERSITY_SCORE': 0.0,
        'GR_CONFIDENCE_SCORE': 0.0, 'GR_GEOLOGIC_RIGOR_SCORE': 0.0,
        'MODEL_COLUMNS': '',
        'PIXEL_MODEL_UPDATE': ''
    }


    # default pixel_model dictionary
    __pixel_model_dict__ = {
        "MODEL_UID": "",
        "CELL_ID": 0,
        "RD_DISC_HC": 0.0,
        "VOL_DISC_HC": 0.0,
        "RD_OUTPUT_HC": 0.0,
        "VOL_OUTPUT_HC": 0.0,
        "RD_YTF_HC": 0.0,
        "VOL_YTF_HC": 0.0,
        "LABEL_PIXEL": 0.0,
        "CLASS_INPUT": 0.0,
        "CLASS_ASSIGNED": 0.0,
        "CLASS_PROBABILITY": 0.0,
        "CLASS_PROB_RISK": 0.0,
        "VOL_OUTPUT_MEAN_HC": 0.0,
        "VOL_OUTPUT_STDEV_HC": 0.0,
        "VOL_OUTPUT_SKEW_HC": 0.0,
        "CLUSTER_RANK": 0.0,
        "PREDICTED_RESOURCE_VOLUME_MMBOE": 0.0,
        "COMMITTEEMEDIAN": 0.0,
        "COMMITTEEMEAN": 0.0,
        "COMMITTEEMAX": 0.0,
        "COMMITTEESTDEV": 0.0,
        "PIXEL_DESIGNATION": 0.0
    }


        # default ds_model dictionary
    __ds_model_dict__ = {
        "headerTag": {
            "model_uid": "",
            "modelName": "",
            "createdOn": "",
            "organization": "",
            "modelAuthor": "",
            "modelType": "",
            "modelAlgorithm": "",
            "modelOutputUnit": "",
            "modelComments": ""
        },
        "modelTag": {
            "itemName": "",
            "createdOn": "",
            "itemFormatType": "",
            "itemVersion": "",
            "itemLocation": ""
        },
        "labelTag": {
            "itemName": "",
            "createdOn": "",
            "itemFormatType": "",
            "itemVersion": "",
            "itemLocation": ""
        },
        "featuresTag": [],
        "metrics": {
            "rmse_pct_train_pixels": 0.0,
            "mae_pct_train_pixels": 0.0,
            "adjusted_r2_train_pixels": 0.0,
            "rmse_pct_test_pixels": 0.0,
            "mae_pct_test_pixels": 0.0,
            "adjusted_r2_test_pixels": 0.0,
            "rmse_pct_holdout_pixels": 0.0,
            "mae_pct_holdout_pixels": 0.0,
            "adjusted_r2_holdout_pixels": 0.0,
            "true_positive_rate": 0.0,
            "true_negative_rate": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "area_under_roc_curve": 0.0,
            "silhouette_coeff": 0.0,
            "calinski_harabasz_idx": 0.0,
            "newman_girvan_mod": 0.0,
            "gr_sufficiency_score": 0.0,
            "gr_density_score": 0.0,
            "gr_diversity_score": 0.0,
            "gr_confidence_score": 0.0,
            "gr_geologic_rigor_score": 0.0
        },
        "stats": {
            "pixel_count_non_label_neg_ytf": 0.0,
            "basin_count_non_label_neg_ytf": 0.0,
            "sum_non_label_neg_ytf": 0.0,
            "pixel_global_vol_ytf_hc": 0.0,
            "basin_global_vol_ytf_hc": 0.0
        }
    }

    __featureTag__ = {
        "itemName": "",
        "createdOn": "",
        "itemFormatType": "",
        "itemVersion": "",
        "itemLocation": ""
    }


    __HEADERTAG__ = [
        'MODEL_UID', 'MODEL_NAME', 'MODEL_CREATION_DATE',
        'MODEL_SOURCE_ORGANIZATION', 'MODEL_AUTHOR', 'MODEL_CLASS', 'MODEL_ALGORITHM_USED',
        'MODEL_OUTPUT_UNIT', 'COMMENTS'
    ]


    __MODELTAG__ = [
        'LOCATION', 'FORMATTYPE'
    ]


    __LABELTAG__ = [
        'LABELNAME', 'LOCATION', 'FORMATTYPE', 'CREATEDON',
        'LASTMODIFIEDON'
    ]


    __METRICS__ = [
        'RMSE_PCT_TRAIN_PIXELS', 'MAE_PCT_TRAIN_PIXELS', 'ADJUSTED_R2_TRAIN_PIXELS', 'RMSE_PCT_TEST_PIXELS',
        'MAE_PCT_TEST_PIXELS', 'ADJUSTED_R2_TEST_PIXELS', 'RMSE_PCT_HOLDOUT_PIXELS', 'MAE_PCT_HOLDOUT_PIXELS',
        'ADJUSTED_R2_HOLDOUT_PIXELS', 'TRUE_POSITIVE_RATE', 'TRUE_NEGATIVE_RATE', 'FALSE_POSITIVE_RATE',
        'FALSE_NEGATIVE_RATE', 'AREA_UNDER_ROC_CURVE', 'SILHOUETTE_COEFF', 'CALINSKI_HARABASZ_IDX',
        'NEWMAN_GIRVAN_MOD', 'GR_SUFFICIENCY_SCORE', 'GR_DENSITY_SCORE', 'GR_DIVERSITY_SCORE',
        'GR_CONFIDENCE_SCORE', 'GR_GEOLOGIC_RIGOR_SCORE'
    ]


    __STATS__ = [
        'PIXEL_COUNT_NON_LABEL_NEG_YTF', 'BASIN_COUNT_NON_LABEL_NEG_YTF', 'SUM_NON_LABEL_NEG_YTF', 'PIXEL_GLOBAL_VOL_YTF_HC',
        'BASIN_GLOBAL_VOL_YTF_HC'
    ]


    #defining the query BHP_JGE.PHASE2_MODEL_CATALOG
    __query_insert_record_model_catalog__ = """
        INSERT INTO BHP_JGE.PHASE2_MODEL_CATALOG (
        MODEL_UID, MODEL_NAME, MODEL_CREATION_DATE, MODEL_SOURCE_ORGANIZATION,
        MODEL_AUTHOR, MODEL_CLASS, MODEL_ALGORITHM_USED, MODEL_OUTPUT_UNIT,
        RMSE_PCT_TRAIN_PIXELS, MAE_PCT_TRAIN_PIXELS, ADJUSTED_R2_TRAIN_PIXELS, RMSE_PCT_TEST_PIXELS,
        MAE_PCT_TEST_PIXELS, ADJUSTED_R2_TEST_PIXELS, RMSE_PCT_HOLDOUT_PIXELS, MAE_PCT_HOLDOUT_PIXELS,
        ADJUSTED_R2_HOLDOUT_PIXELS, TRUE_POSITIVE_RATE, TRUE_NEGATIVE_RATE, FALSE_POSITIVE_RATE,
        FALSE_NEGATIVE_RATE, AREA_UNDER_ROC_CURVE, SILHOUETTE_COEFF, CALINSKI_HARABASZ_IDX,
        NEWMAN_GIRVAN_MOD, PIXEL_COUNT_NON_LABEL_NEG_YTF, BASIN_COUNT_NON_LABEL_NEG_YTF, SUM_NON_LABEL_NEG_YTF,
        PIXEL_GLOBAL_VOL_YTF_HC, BASIN_GLOBAL_VOL_YTF_HC,
        COMMENTS,
        GR_SUFFICIENCY_SCORE, GR_DENSITY_SCORE, GR_DIVERSITY_SCORE, GR_CONFIDENCE_SCORE,
        GR_GEOLOGIC_RIGOR_SCORE,
        MODEL_COLUMNS, PIXEL_MODEL_UPDATE)
        VALUES (
        :1, :2, TO_DATE(:3, 'YYYY-MM-DD'), :4,
        :5, :6, :7, :8,
        :9, :10, :11, :12,
        :13, :14, :15, :16,
        :17, :18, :19, :20,
        :21, :22, :23, :24,
        :25, :26, :27, :28,
        :29, :30,
        :31,
        :32, :33, :34, :35,
        :36,
        :37, :38
        )
    """


    def get_query_insert_model_catalog(self):
        return self.__query_insert_record_model_catalog__


    #defining the query BHP_JGE.PHASE2_PIXEL_MODELS_VERT
    __query_insert_record_pixel_model__ = """
        INSERT INTO BHP_JGE.PHASE2_PIXEL_MODELS_VERT (
            MODEL_UID, CELL_ID, RD_DISC_HC, VOL_DISC_HC, RD_OUTPUT_HC, VOL_OUTPUT_HC,
            RD_YTF_HC, VOL_YTF_HC, LABEL_PIXEL, CLASS_INPUT, CLASS_ASSIGNED, CLASS_PROBABILITY,
            CLASS_PROB_RISK, VOL_OUTPUT_MEAN_HC, VOL_OUTPUT_STDEV_HC, VOL_OUTPUT_SKEW_HC, CLUSTER_RANK, PREDICTED_RESOURCE_VOLUME_MMBOE,
            COMMITTEEMEDIAN, COMMITTEEMEAN, COMMITTEEMAX, COMMITTEESTDEV, PIXEL_DESIGNATION
        )
        VALUES (
            :1, :2, :3, :4, :5, :6,
            :7, :8, :9, :10, :11, :12,
            :13, :14, :15, :16, :17, :18,
            :19, :20, :21, :22, :23
        )
    """


    def get_query_insert_pixel_model(self):
        return self.__query_insert_record_pixel_model__


    def get_new_model_uid(self, last_model_uid, current_year=datetime.datetime.now().year):
        """
        create an incremental of a new model uid
        sample:
        latest_model_uid = 'M2021_11' --> 'M2021_12' or 'M2022_1'
        """
        # pattern to extract year from last_model_uid
        pattern = re.compile(r'M(\d+)\_(\d+)$')
        match = re.split(pattern, last_model_uid)
        if match:
            last_model_year = int(match[1])
            last_consecutive = int(match[2])
        if last_model_year < current_year:
            last_model_year = current_year
            last_consecutive = 1
        else:
            last_consecutive += 1
        return 'M' + str(last_model_year) + '_' + str(last_consecutive)


    def get_last_model_uid_dynamo(self):

        # connection setup
        connection_name = self.__dyn_connection_name__

        tableName = self.get_dynamo_table_name()

        client = dataiku.api_client()
        info = dataikuapi.dss.admin.DSSConnection(name=connection_name, client=client).get_info()
        aws_creds = info['params']
        aws_access_key_id = aws_creds['accessKey']
        aws_secret_access_key = aws_creds['secretKey']
        region = aws_creds['regionOrEndpoint']

        # dynamoDB connection
        dynamodb = boto3.resource('dynamodb', region_name=region,
                                              aws_access_key_id=aws_access_key_id,
                                              aws_secret_access_key=aws_secret_access_key)
        table = dynamodb.Table(tableName)

        scan_args = {
            'ProjectionExpression': 'model_uid'
        }

        response = table.scan(**scan_args)

        if response['ResponseMetadata']['HTTPStatusCode'] not in range(200, 300):
            return ''

        data = response['Items']
        dm = pd.DataFrame(data)
        dm['model_uid_sort'] = dm.apply(lambda x: str(x.model_uid).split('_')[0] + str(x.model_uid).split('_')[1].zfill(5), axis=1)
        dm.sort_values(by=['model_uid_sort'], ascending=False, inplace=True)
        return dm.head(1)['model_uid'].values[0]


    def prepare_model_catalog_df(self, model_uid, df_model_report_csv):
        # create dataframe (with one line) for the model_report table: PHASE2_MODEL_CATALOG
        df_model_report_csv.columns = df_model_report_csv.columns.str.upper()
        model_report_oneline = dict({'MODEL_UID': model_uid},
                                    **dict(zip(df_model_report_csv['COLUMN_NAME'].values,
                                               df_model_report_csv['COLUMN_VALUE'].values)))
        for k, v in model_report_oneline.items():
            if k == 'MODEL_CREATION_DATE':
                month, day, year = v.split('/')
                v = str(year)+'-'+str(month)+'-'+str(day)
            self.__model_report_dict__[k] = v
        df = pd.DataFrame(self.__model_report_dict__, index=[0])
        df['PIXEL_MODEL_UPDATE'] = 'Complete'

        return df


    def prepare_pixel_model_df(self, model_uid, df_pixel_models_csv):

        df_pixel_models_csv.columns = df_pixel_models_csv.columns.str.upper()
        cols = list(df_pixel_models_csv.columns)

        if (('LATITUDE' in cols) & ('LONGITUDE' in cols)):
            df_pixel_models_csv.sort_values(
                by=['LATITUDE', 'LONGITUDE'], axis=0, ignore_index=True, inplace=True, ascending=True)

        # add CELL_ID if CELL_ID doesn't exist -- then re-arrange columns
        if 'CELL_ID' not in cols:
            df_pixel_models_csv['CELL_ID'] = [
                i for i in df_pixel_models_csv.index + 1]

        df_pixel_models_csv.sort_values(
            by=['CELL_ID'], axis=0, ignore_index=True, inplace=True, ascending=True)

        df_pixel_models_csv['MODEL_UID'] = model_uid

        # select only columns CELL_ID + all others but LATITUDE and LONGITUDE
        pxl_mdls_cols = [col for col in df_pixel_models_csv.columns if not col in [
            'LATITUDE', 'LONGITUDE']]
        df_pixel_models_csv = df_pixel_models_csv[pxl_mdls_cols].copy()

        for k in self.__pixel_model_dict__.keys():
            self.__pixel_model_dict__[k] = None
            if k in df_pixel_models_csv.columns:
                self.__pixel_model_dict__[k] = df_pixel_models_csv[k].values

        df = pd.DataFrame(self.__pixel_model_dict__)
        return df


    def InConverter(self, value):
        # or whatever is needed to convert from numpy.int64 to an integer
        return int(value)


    def InputTypeHandler(self, cursor, value, num_elements):
        if isinstance(value, np.int64):
            return cursor.var(int, arraysize=num_elements, inconverter=InConverter)


    def insert_records(self, df, query_insert_record):
        """
        insert records in MODEL_CATALOG
        """
        status = False
        records = df.to_records(index=False)
        data = list(records)

        user, password, dsn = self.get_oracle_credentials()

        try:
            connection = cx_Oracle.connect(user=user,
                                           password=password,
                                           dsn=dsn,
                                           encoding="UTF-8")
            cursor = connection.cursor()

            # inserting the rows
    #         cursor.inputtypehandler = InputTypeHandler
            cursor.executemany(query_insert_record, data)
            connection.commit()
            status = True

        except cx_Oracle.Error as error:
            print("Oracle-Error-Code:", error)

        finally:
            cursor.close()
            connection.close()
        return status


    def insert_records_PIXEL(self, df, query_insert_record):
        """
        insert records in table PIXEL_MODELS
        """
        df = df.replace(np.nan, 0.0)
        df['CELL_ID'] = df['CELL_ID'].astype(str)

        status = False
        records = df.to_records(index=False)
        data = list(records)

        user, password, dsn = self.get_oracle_credentials()

        try:
            connection = cx_Oracle.connect(user=user,
                                           password=password,
                                           dsn=dsn,
                                           encoding="UTF-8")
        except cx_Oracle.DatabaseError as error:
            print('There is an error in the Oracle database:', error)

        else:

            try:

                cursor = connection.cursor()
                query = query_insert_record

                # inserting the rows
    #             cursor.inputtypehandler = InputTypeHandler
                cursor.executemany(query, data)
                status = True

            except cx_Oracle.DatabaseError as error:
                print('There is an error in the Oracle database:', error)

            except Exception as error:
                print('Error: ' + str(error))

            finally:
                if cursor:
                    connection.commit()
                    cursor.close()

        finally:
            if connection:
                connection.close()

        return status


    def coerce_df_columns_to_numeric(self, df, column_list):
        df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
        return df


    def coerce_dataframe(self, df, df_pixel_columns):
        column_list = list(df.columns[8:30]) + list(df.columns[31:36])
        df = self.coerce_df_columns_to_numeric(df, column_list)
        df = df.replace(np.nan, 0.0)
        df['MODEL_COLUMNS'] = ', '.join(df_pixel_columns[2:])
        return df


    def get_data(self, inputName=None, capitalize=True):

        if inputName is None:
            return pd.DataFrame()

        df = pd.read_csv(inputName)
        if capitalize:
            # capitalize all column names
            df.columns = df.columns.str.upper()
            return df


    def get_dictionary_type(self, dict_type='ds_model'):
        dictionary_types = ['ds_model', 'model_report', 'pixel_model']
        if dict_type not in dictionary_types:
            return {}
        if dict_type == 'ds_model':
            return self.__ds_model_dict__
        if dict_type == 'model_report':
            return self.__model_report_dict__
        if dict_type == 'pixel_model':
            return self.__pixel_model_dict__


    def get_featureTag(self):
        return self.__featureTag__


    def prepare_ds_model_dynamo(self, df):

        metrics = dict()
        for k, v in dict(zip(self.__METRICS__, self.get_dictionary_type()['metrics'].keys())).items():
            t = type(self.get_dictionary_type()['metrics'][v])
            metrics[v] = t(df[k].values[0])

        stats = dict()
        for k, v in dict(zip(self.__STATS__, self.get_dictionary_type()['stats'].keys())).items():
            t = type(self.get_dictionary_type()['stats'][v])
            stats[v] = t(df[k].values[0])

        hdrTag = dict(zip(self.get_dictionary_type()['headerTag'].keys(), df[self.__HEADERTAG__].values[0].tolist()))
        mdlTag = {'modelTag': self.get_dictionary_type()['modelTag']}
        lblTag = {'labelTag': self.get_dictionary_type()['labelTag']}
        feaTag = {'featuresTag': self.get_dictionary_type()['featuresTag']}
        metTag = {'metrics': metrics}
        staTag = {'stats': stats}

        toDynamo = dict(**hdrTag, **mdlTag, **lblTag, **feaTag, **metTag, **staTag)
        for k, v in toDynamo['metrics'].items():
            if np.isnan(v):
                toDynamo['metrics'][k] = 0.0

        for k, v in toDynamo['stats'].items():
            if np.isnan(v):
                toDynamo['metrics'][k] = 0.0
        self.__ds_model_dict__ = json.loads(json.dumps(toDynamo), parse_float=Decimal)
        return


    def update_ds_model_dynamo(self, df, tag="modelTag"):
        """
        update modelTag, labelTag, and featuresTag
        input:
            item: json data to update with new data
            df:   dataframe that contains additional data
            tag:  key to update ["modelTag", "labelTag", "featuresTag"]
        """
        if tag not in {"modelTag", "labelTag", "featuresTag"}:
            return self.get_dictionary_type()

#             __featureTag__ = {
#                 "itemName": "",
#                 "createdOn": "",
#                 "itemFormatType": "",
#                 "itemVersion": "",
#                 "itemLocation": ""
#             }

        if (tag == "modelTag") | (tag == "labelTag"):
            targetKeys = self.get_dictionary_type()[tag].keys()
            targetVals = df.loc[df['ITEMTAGTYPE'] == tag, :].values.tolist()[0][1:-1]
            self.get_dictionary_type()[tag] = dict(zip(targetKeys, targetVals))

        elif (tag == "featuresTag"):
            for item in df.loc[df['ITEMTAGTYPE'] == 'featureTag', :].itertuples(index=False, name=None):
                targetKeys = self.get_featureTag().keys()
                targetVals = list(item)[1:-1]
                self.get_dictionary_type()[tag].append(dict(zip(targetKeys, targetVals)))

        return


    def put_item_toDynamoDB(self, item):

        # connection setup
        connection_name = self.get_dynamo_connection_name()
        tableName = self.get_dynamo_table_name()

        client = dataiku.api_client()
        info = dataikuapi.dss.admin.DSSConnection(name=connection_name, client=client).get_info()
        aws_creds = info['params']
        aws_access_key_id = aws_creds['accessKey']
        aws_secret_access_key = aws_creds['secretKey']
        region = aws_creds['regionOrEndpoint']

        # dynamoDB connection
        dynamodb = boto3.resource('dynamodb', region_name=region,
                                              aws_access_key_id=aws_access_key_id,
                                              aws_secret_access_key=aws_secret_access_key)
        table = dynamodb.Table(tableName)

        response = table.put_item(Item=item)

        if response['ResponseMetadata']['HTTPStatusCode'] not in range(200, 300):
            return False

        return True

