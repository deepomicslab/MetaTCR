# metadata_clean — 精简后的下游 metadata

生成日期:2026-08-23。这是 `MetaTCR_v3/data/metadata/` 的**精简副本**(源目录未改动)。只收录**被实际脚本用过的数据集**,并删掉了从不被读取的技术/溯源冗余列。

## 收录范围
- **20 个数据集 CSV** + `datasets_type.csv`(config,3 列全用,原样保留)。
- **Genolet2023** 单独收录:因它无脚本读取,按要求**只删统一删的那几列**(此处为 `sample_type/assay/replicate`,无 umi),其余信息(含 `species/comparison_group/sequencing_instrument/geo_accession`)**全部保留**——与其它文件不同,不对它做溯源列精简。
- **排除**:仅 `Chan2026`。原目录的 `achieved/` 归档、旧 `README.md` 也不收。

## 清理策略
- **强制删除**:`umi`、`assay`、`sample_type`、`replicate`。
- **删除从不被读取的纯技术/溯源列**:`species`、`comparison_group`、`sequencing_instrument`、`geo_accession`、`source_release`、`source_institution`、`source_publication`、`release_publication`、`release_accession`、`primary_biological_view`,以及 ImmuneCODE 的 COVID 明细标记(`virus_disease`、`covid_category`、`covid_diagnosis`、`covid_exposed`、`covid_recovered`、`hospitalized`、`icu_admit`、`status`、`who_ordinal_scale`、`death`、`pcr_positive`)。
- **一定保留**:所有**被脚本用过的列**(已用断言强制校验,全部通过);外加人口学/临床信息 `age`、`sex`、`ancestry`、`ethnicity`、`cancer_type`、`disease_stage`,以及 `protocol_family`(它在 `datasets_type.csv` 里被脚本使用,此处也一并保留作为描述)。

## 每个文件的精简结果

| 文件 | 列数 (原→精简) | 删除的列 |
|--|--|--|
| Barennes2021.csv | 17 → 13 | sample_type, assay, umi, replicate |
| Emerson2017.csv | 10 → 7 | sample_type, assay, umi |
| Genolet2023.csv | 15 → 12 | sample_type, assay, replicate *(仅统一删的列;species/comparison_group/sequencing_instrument/geo_accession 等均保留)* |
| Heather2015.csv | 12 → 9 | sample_type, assay, umi |
| Huth2019.csv | 9 → 6 | sample_type, assay, umi |
| Huuhtanen2022.csv | 14 → 11 | sample_type, assay, umi |
| ImmuneCODE.csv | 26 → 11 | sample_type, assay, umi, virus_disease, covid_category, covid_diagnosis, covid_exposed, covid_recovered, hospitalized, icu_admit, status, who_ordinal_scale, death, pcr_positive, source_release |
| Liu2019.csv | 9 → 6 | sample_type, assay, umi |
| Nair2025.csv | 14 → 11 | sample_type, assay, umi |
| Rawat2026.csv | 13 → 10 | sample_type, assay, umi |
| Robert2014.csv | 12 → 9 | sample_type, assay, umi |
| Sherwood2015.csv | 13 → 10 | sample_type, assay, umi |
| Snyder2017.csv | 13 → 10 | sample_type, assay, umi |
| Valpione2020.csv | 13 → 10 | sample_type, assay, umi |
| Visvabharathy2023.csv | 11 → 8 | sample_type, assay, umi |
| Wang2022.csv | 14 → 11 | sample_type, assay, umi |
| Weber2018.csv | 13 → 10 | sample_type, assay, umi |
| Wright2026.csv | 18 → 15 | sample_type, assay, umi |
| Ye2020.csv | 9 → 6 | sample_type, assay, umi |
| Zaslavsky2025.csv | 18 → 12 | assay, primary_biological_view, source_institution, source_publication, release_publication, release_accession |
| datasets_type.csv | 3 → 3 | (不变) |

## 保障
- 数据值未改动,仅按列删除;字符串按原样保留(未做数值/NA 转换)。
- 每个数据集「被脚本用过的列」在写出前均通过断言校验,确保不被误删。
- 列使用的完整证据见上一级目录的 `metadata_column_usage.md`。
