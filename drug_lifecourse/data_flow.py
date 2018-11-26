__author__ = 'thor'

from numpy import *

import ut as ms
from ut.util.data_flow import DataFlow
import ut.pfile.accessor as pfile_accessor
import pandas as pd
import ut.daf.manip as daf_manip
from ut.util.pstore import MyStore
import ut.util.log as util_log
import ut.util.pobj as util_pobj
import ut.pdict.get as pdict_get
import ut.daf.ch as daf_ch
import ut.daf.manip
from misc.drug_lifecourse.constants import binarization_map

class DrugLifeCourse(DataFlow):
    def __init__(self, **kwargs):
        params = dict()
        params['facc'] = pfile_accessor.for_local('socio/drug_lifecourse')
        params['yearly_csv_file'] = params['facc']('ODUS_Yearly.csv')
        params['store'] = MyStore(params['facc']('lifecourse.h5'))
        params['data_dependencies'] = {
            'raw_life_course_data': 'yearly_csv_file',
            'id_year_of_birth': 'raw_life_course_data',
            'drug_use_data': ['raw_life_course_data', 'id_year_of_birth'],
            'drug_use_and_social_binary_data': ['raw_life_course_data', 'id_year_of_birth']
        }
        # data makers
        #data_makers_whose_method_name_is_the_name_of_the_data = [
        #    'raw_hotel_data', 'processed_hotel_data', 'geo__id_city_country',
        #    'poi_data_source_filename', 'raw_poi_data', 'processed_poi_data',
        #    'preped_poi_data', 'poi__poi_lat_lon', 'poi__name_and_city',
        #    'poi__name_and_city_kw', 'poi__bmms',
        #    'templates__ads', 'templates__keywords'
        #]
        params['data_makers'] = {k: params[k] for k in params['data_dependencies'].keys() if k in params.keys()}
        # data storers
        params['data_storers'] = {
            'raw_hotel_data': self.put_in_store
        }
        kwargs = dict(kwargs, **params)
        super(DrugLifeCourse, self).__init__(**kwargs)
        self.variable_map = {
            'druguse_1': 'alc',
            'druguse_2': 'tob',
            'druguse_3': 'mar',
            'druguse_4': 'hal',
            'druguse_5': 'prp',
            'druguse_6': 'coc',
            'druguse_7': 'crk',
            'druguse_8': 'her',
            'druguse_9': 'amp',
            'druguse_10': 'met',
            'famrole_3': 'parent'
        }
        self.drugs = ['alc', 'tob', 'mar', 'hal', 'prp', 'coc', 'crk', 'her', 'amp', 'met']
        self.col_order_01 = ['id', 'age'] + self.drugs + ['year']

    @staticmethod
    def raw_life_course_data(yearly_csv_file, **kwargs):
        return pd.read_csv(yearly_csv_file)

    @staticmethod
    def id_year_of_birth(raw_life_course_data, **kwargs):
        d = raw_life_course_data[['id', 'year']].groupby('id').min()
        return daf_ch.ch_col_names(d, 'yob', 'year').reset_index()

    def drug_use_data(self, raw_life_course_data, **kwargs):
        cols = ['id', 'year',
                'druguse_1', 'druguse_2', 'druguse_3', 'druguse_4', 'druguse_5',
                'druguse_6', 'druguse_7', 'druguse_8', 'druguse_9', 'druguse_10']
        df = daf_manip.filter_columns(raw_life_course_data, cols)
        df = self.add_age(df)
        df = self.process_cols(df)
        return df

    def drug_use_and_social_binary_data(self, raw_life_course_data, **kwargs):
        cols = ['id', 'year',
                'druguse_1', 'druguse_2', 'druguse_3', 'druguse_4', 'druguse_5',
                'druguse_6', 'druguse_7', 'druguse_8', 'druguse_9', 'druguse_10',
                'prison', 'sexint', 'condom', 'famrole_3', 'famrole_4', 'famrole_5']
        # famrole_cols = ['famrole_1', 'famrole_2', 'famrole_3', 'famrole_4',
        #                 'famrole_5', 'famrole_6', 'famrole_7']
        df = daf_manip.filter_columns(raw_life_course_data, cols)
        df['partner_or_spouse'] = any(df[['famrole_4', 'famrole_5']], axis=1).astype(int)
        df = ms.daf.manip.rm_cols_if_present(df, ['famrole_4', 'famrole_5'])
        # df['family_role'] = any([df[c] for c in famrole_cols])
        # del df[famrole_cols]
        df = self.binarize(df)
        df = self.add_age(df)
        df = self.process_cols(df)
        return df


    ######################### UTILS #################################################
    @staticmethod
    def add_age(df):
        if 'yob' not in df.columns:
            remove_yob = True
            id_yob = DrugLifeCourse.id_year_of_birth(df)
            df = df.merge(id_yob)
        else:
            remove_yob = False
        df['age'] = df['year'] - df['yob']
        if remove_yob:
            df = daf_manip.rm_cols_if_present(df, 'yob')
        return df

    def ch_col_names(self, df):
        return daf_ch.ch_col_names(df, self.variable_map.values(), self.variable_map.keys())

    def order_cols(self, df):
        return daf_manip.reorder_columns_as(df, self.col_order_01)

    def process_cols(self, df):
        return self.order_cols(self.ch_col_names(df))

    @staticmethod
    def binarize(df):
        zero_column = zeros((len(df), 1))
        for c in df.columns:
            if c in binarization_map.keys():
                binmap = binarization_map[c]
                df[binmap['bin_name']] = zero_column
                df[binmap['bin_name']][df[c].isin(binmap['ifin'])] = 1
                del df[c]
        return df