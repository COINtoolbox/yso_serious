# Copyright 2022 COIN and Fink
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

import setuptools

setuptools.setup(
    name='yso_serious',
    version='0.1',
    packages=setuptools.find_packages(),
    py_modules=['yso_serious'],
    scripts=['yso_serious/scripts/train_iso_forest.py'],
    url='https://github.com/COINtoolbox/yso_serious',
    license='MIT',
    author='Emille E. O.Ishida',
    author_email='emille@cosmostatistics-initiative.org',
    description='Fink-COIN module for YSO-dippers'
)