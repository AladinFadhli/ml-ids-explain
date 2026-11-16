# Data — not versioned

1. Download **MachineLearningCSV.zip** from the official CIC-IDS2017 page:
   https://www.unb.ca/cic/datasets/ids-2017.html (form, then download link).
2. Unzip the 8 CSV files (`Monday-WorkingHours.pcap_ISCX.csv` … `Friday-…`) into `data/raw/`.
3. On Colab: put them in a Drive folder and set `IDS_DATA_DIR` in the first notebook cell.

The corrected 2021 release by Engelen et al. (https://downloads.distrinet-research.be/WTMC2021)
also works: same loader; flows labelled "Attempted" are removed and counted.

Nothing in `data/raw/` is committed (see `.gitignore`).
