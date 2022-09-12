# Copyright 2022
# Author: Emille Ishida
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

from yso_serious.utils import *

def main():
    
    filter_names = ['g', 'r']
    
    ############################################################################
    ################  User choices    ##########################################
    
    fname_dippers = '../data/light_curves.csv'       # light curve file for dippers
    npoints = 5                                  # min number of points per filter
    
    fname_other = '/media2/YSO_Spur/data/gaia_allcat_lc_features_10pc.parquet'   
    fname_yso = '/media2/YSO_Spur/fink_YSO/lit_ysos_for_coin_fink_match_features.parquet'
    
    train_model = False
    fname_model = '../../models/yso_serious_model.pkl'
    
    fname_scores = '../data/scores.pkl'
    
    #############################################################################

    # read dippers light curve from data taken from api
    use_ids_dippers, pdf_dippers = selection_cuts(fname_dippers, npoints=npoints)
    features_dippers = calculate_features_from_api(use_ids_dippers, lc_pdf=pdf_dippers)
    
    # read other and yso features from parquet file
    features_other = build_features_from_parquet(fname_other)
    features_yso = build_features_from_parquet(fname_yso)
    
    # fit isolaton forest
    if train_model:
        clf = IsolationForest().fit(features_dippers.values[:,1:])
        if isinstance(fname_model, str):
            pickle.dump(clf, open(fname_model, 'wb'))
    else:
        clf = pickle.load(open(fname_model, 'rb'))
        
    # calculate and save scores to file
    scores = {}    
    scores['dippers'] = clf.score_samples(features_dippers.values[:,1:])
    scores['other'] = clf.score_samples(features_other.values[:,1:])
    scores['yso'] = clf.score_samples(features_yso.values[:,1:])
    
    f = open(fname_scores,"wb")
    pickle.dump(scores,f)
    f.close()
    
if __name__ == '__main__':
    main()
