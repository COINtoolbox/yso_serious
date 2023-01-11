# Copyright COIN and Fink 2022 
# Author: Emille Ishida and Víctor Almendros-Abad
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/mit-license.php
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import pickle
import numpy as np

from copy import deepcopy
from sklearn.ensemble import IsolationForest
from fink_science.ad_features.processor import create_extractor

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    'build_header',
    'selection_cuts',
    'mag_asymmetry',
    'calculate_features_from_api',
    'build_features_from_parquet'
]

def build_header():
    """
    Build header for light curve parameters data frame.
    
    Returns
    -------
    header:
        list of str: header of light curve parameter data frame.
    """
    
    column_names = create_extractor().names
    header = ['objectId'] + [item + '_g' for item in column_names] +  \
             ['asymmetry_g'] + [item + '_r' for item in column_names] + \
             ['asymmetry_r'] 
    
    return header


def selection_cuts(fname_lcs: str, npoints=5):
    """
    Apply quality selection cuts on light curves.
    
    Parameters
    ----------
    fname_lcs: str
        Path to light curve file.
    npoints: int (optional)
        Minimum number of points required per filter.
        Default is 5.
        
    Returns
    -------
    list: 
        objectId surviving selection cuts.     
    pd.DataFrame:
        Complete light curves from api.
    """
    # read light curves
    pdf = pd.read_csv(fname_lcs, index_col=False)
 
    # get unique objectIds
    unique_ids = np.unique(pdf['i:objectId'].values)

    use_ids = []

    for name in unique_ids:
        # count number of photometric points per band    
        flag_g = pdf[pdf['i:objectId'] == name]['i:fid'].values == 1
        flag_r = pdf[pdf['i:objectId'] == name]['i:fid'].values == 2
    
        if sum(flag_g) >= npoints and sum(flag_r) >= npoints:
            use_ids.append(name)
            
    return use_ids, pdf


def mag_asymmetry(y: np.array):
    """Calculate magnitud asymmetry.
    
    Parameters
    ----------
    y: np.array
        Observed magnitudes.
        
    Returns
    -------
    float: 
        Magnitude asymmetry.
    """

    p10, median, p90 = np.percentile(y,10),np.percentile(y,50),np.percentile(y,90)
    std = np.nanstd(y)
    M = (np.mean([p10,p90]) - median) / std

    return M


def calculate_features_from_api(ids: list, lc_pdf: pd.DataFrame):
    """
    Calculate all light curve features from ad_features + mag asymmetry.
    
    Parameters
    ----------
    ids: list
        List of objectId surviving selection cuts.
    lc_df: pd.DataFrame
        Light curves.
        
    Returns
    -------
    pd.DataFrame
        All light curve features from ad_features + mag asymmetry.
    """
    
    extractor = create_extractor()
    
    data_list = []

    for obj_id in ids:

        # separate photometric points for 1 object
        lc_points = lc_pdf[lc_pdf['i:objectId'] == obj_id]

        features = [obj_id]

        for band in range(1,3):
            flag_filter = lc_points['i:fid'].values == band
            indx = np.argsort(lc_points[flag_filter]['i:jd'].values)

            # extract feature from light curve package
            results = list(extractor(
                              lc_points[flag_filter]['i:jd'].values[indx],
                              lc_points[flag_filter]['i:magpsf'].values[indx],  
                              lc_points[flag_filter]['i:sigmapsf'].values[indx]))
            
            M = mag_asymmetry(lc_points[flag_filter]['i:magpsf'].values)

            features = features + results + [M]
            
        data_list.append(features)

    features = pd.DataFrame(data_list, columns=build_header())
    features.dropna(inplace=True)
    
    return features


def build_features_from_parquet(fname: str):
    """
    Build all light curve features from ad_features + mag asymmetry.
    
    Parameters
    ----------
    fname: str
        Name of parquet file with already extracted features.
    
    Returns
    -------
    pd.DataFrame
        All light curve features from ad_features + mag asymmetry.
    """
    
    # read data
    features = pd.read_parquet(fname)
    
    # remove Nones
    flag1 = [features['lc_features'][i]['1'] != None for i in range(features.shape[0])]
    flag2 = [features['lc_features'][i]['2'] != None for i in range(features.shape[0])]
    flag3 = np.logical_and(flag1, flag2)

    data_2bands = deepcopy(features[flag3].drop_duplicates(subset='objectId', keep='first'))

    data_list_other = []

    features_id = data_2bands['lc_features'].keys()

    for i in range(data_2bands.shape[0]):
        line = [data_2bands.iloc[i]['objectId']]

        for k in ['1', '2']:
            for name in create_extractor().names:
                line.append(data_2bands['lc_features'][features_id[i]][k][name])
            
            flag_filter = data_2bands.iloc[i]['cfid'] == int(k)
            y = data_2bands.iloc[i]['dcmag'][flag_filter]
            line.append(mag_asymmetry(y))
            
        data_list_other.append(line)
        
    data_final = pd.DataFrame(data_list_other, columns=build_header())
    data_final.dropna(inplace=True)
    
    return data_final


def main():    
    return None
    
if __name__ == '__main__':
    main()
