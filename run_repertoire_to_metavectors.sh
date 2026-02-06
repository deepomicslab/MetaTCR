## for melanoma cohort datasets 
python step2.0.dataset_to_meta_matrix.py --unlabeled_dir data/repertoire_data/Robert2014 --dataset_name Robert2014 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Huuhtanen2022 --dataset_name Huuhtanen2022 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Valpione2020 --Shi2017dataset_name Valpione2020 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Weber2018 --dataset_name Weber2018 && \

python step2.0.dataset_to_meta_matrix.py  --pos_dir data/repertoire_data/Liu2019/SLE --neg_dir data/repertoire_data/Liu2019/Control --dataset_name Liu2019 && \
python step2.0.dataset_to_meta_matrix.py  --pos_dir data/repertoire_data/Tcrbv4/PBMC_cancer --neg_dir data/repertoire_data/Tcrbv4/PBMC_healthy --dataset_name Tcrbv4 && \

python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Tcrbv4/PBMC_cancer  --dataset_name Tcrbv4_Tumor && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Tcrbv4/PBMC_healthy --dataset_name Tcrbv4_Normal && \

## wang2022
python step2.0.dataset_to_meta_matrix.py  --pos_dir data/repertoire_data/Wang2022/Tumor --neg_dir data/repertoire_data/Wang2022/Normal --dataset_name Wang2022 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Dewitt2015 --dataset_name Dewitt2015  ## 18PBMC

## cancer
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Formenti2018 --dataset_name Formenti2018 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/TRACERx --dataset_name TRACERx && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Yan2019 --dataset_name Yan2019 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Shi2017 --dataset_name Shi2017 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Jia2018 --dataset_name Jia2018 && \ ## tissue samples
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Snyder2017 --dataset_name Snyder2017


python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/ImmuneCODE --dataset_name ImmuneCODE && \

## 202508
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Martinez2025 --dataset_name Martinez2025 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/Link2023 --dataset_name Link2023 && \
python step2.0.dataset_to_meta_matrix.py  --pos_dir data/repertoire_data/Nair2025/Tumor --neg_dir data/repertoire_data/Nair2025/Normal --dataset_name Nair2025 && \
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir data/repertoire_data/MDanderson2019 --dataset_name MDAnderson2019

## multi platform  ## 	Genolet2023
python step2.0.dataset_to_meta_matrix.py  --unlabeled_dir /home/grads/miaozhhuo2/projects/TCRseq_data/platform_dup_data/GSE225984_filt_full/  --dataset_name Genolet2023