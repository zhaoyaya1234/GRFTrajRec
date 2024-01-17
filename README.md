### GRFTrajRec: A Graph-based Representation Framework for Trajectory Recovery via Spatiotemporal Interval-Informed Seq2Seq

#### Anonymous Submission with Paper ID  5300 .

### 1 Requirements

* `python==3.7.16`

* `torch==1.13.1+cu117`

* `dgl==0.6.1`

* `dgl-cu101==0.8.0`

* `geopy==2.4.0`

* `numpy==1.21.5`

* `pandas==1.3.5`

* `rtree==1.0.1  `

* `scikit-learn==1.0.2`

* `torch-geometric==2.3.1`

* `transformers==4.21.2`

* `networkx==2.6.3  `

* `seaborn==0.12.2`

### 2 Data

Porto :the public trajectory dataset https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.

YanCheng: the private dataset.

Road Network: from  the publicly available road network data sites https://www.openstreetmap.org/.

### 3 Metrics

* The accuracy, recall, precision, f1 score of the road segments recovered.
*  The RN\_MAE,RN\_RMSE of distance on road network, as well as MAEand RMSE of the euclidean distance between predicted and actual points to evaluate the performances of different models.

### 4  Usage

* 4.1  data_prepare  ./GRFTrajRec/data_preparation_before_model/

```
1.1  data_prepare_Porto.ipynb : Preprocessing the Porto trajectory and road network data for usage in the model 
1.2 data_prepare_YanCheng.ipynb: Preprocessing the YanCheng trajectory and road network data for usage in the model 
1.3 Generating the Porto data  for usage in the model :
nohup python -u data_generate.py    --city Porto  >data_generate_yancheng.out  2>&1 &
1.4 Generating the YanCheng data  for usage in the model :
nohup python -u data_generate.py    --city yancheng  >data_generate_yancheng.out  2>&1 &
```

* 4.2  Training & testing :   (For further experimentation, you can also set the `keep_ratio` to 0.0625 and choose either `block` or `uniform` for the `ds_type`)

```
nohup python -u muti_main.py    --i 0    --city Porto   --hid_dim 512   --epochs 30    --keep_ratio  0.125   --ds_type random     --test_code  >test_Porto.out 2>&1 &

nohup python -u  muti_main.py   --i 0  --city  yancheng  --hid_dim 128  --epochs  35     --keep_ratio  0.125  --ds_type random --test_code >test_yancheng.out 2>&1 &
```

* 4.3  Ablation Study:  (Using Porto dataset as an example)

```
nohup python -u muti_main.py    --i  0  --city Porto  --hid_dim  512   --epochs  30   --keep_ratio 0.125 --ds_type random       --test_code  >test_Porto.out 2>&1 &

nohup python -u muti_main.py    --i 0  --city Porto  --hid_dim  512  --epochs   30     --keep_ratio 0.125 --ds_type random      --Add_Graph_Representation_flag   --test_code  >test_Porto_Graph_Representation.out 2>&1 &

nohup python -u muti_main.py    --i  0   --city Porto  --hid_dim   512   --epochs   30   --keep_ratio 0.125 --ds_type random      --Add_change_GNN_flag  --test_code  >test_Porto_Add_change_GNN.out 2>&1 &

nohup python -u muti_main.py    --i  0   --city Porto  --hid_dim  512   --epochs  30   --keep_ratio 0.125 --ds_type random    --Add_Traj_Representation_flag      --test_code  >test_Porto_Traj_Representation_flag.out 2>&1 &

nohup python -u muti_main.py    --i 0 --city Porto  --hid_dim   512   --epochs   30   --keep_ratio 0.125 --ds_type random   --Add_transformer_ST_flag   --test_code  >test_Porto_Add_transformer_ST_flag.out 2>&1 &

nohup python -u muti_main.py    --i 0  --city Porto  --hid_dim 512  --epochs   30   --keep_ratio 0.125 --ds_type random    --Add_feature_differences_flag   --test_code  >test_Porto_feature_differences_flag.out 2>&1 &
```

