# CT-FM Data Download Guide

This guide details the steps to download and process the CT scan data used for both pre-training and downstream tasks in the CT-FM study. All datasets are publicly available, ensuring that every element of our study can be reproduced.

## Pre-training Data

For our pre-training experiments, we utilize 148,394 CT scans from the Imaging Data Commons (IDC). Follow the steps below to obtain and prepare the exact dataset used in our study.

### 1. Run the Data Query in BigQuery

Execute the provided SQL query on Google BigQuery to filter for CT scans that meet our quality constraints. The query performs necessary quality checks on each scan.

- Query file: [query.sql](../../notebooks/data-download/query.sql)

Running this query returns a table with CT scan records that satisfy our criteria. We then convert these query results to a manifest file that can be used to download the data

### 2. Generate a Download Manifest
This has already been done so you can skip to the next step if you don't want to know how!

After reviewing the query results, use the Jupyter Notebook to create a manifest file. This manifest lists every DICOM file that needs to be downloaded.

- Manifest creation notebook: [prepare_download.ipynb](../../notebooks/data-download/prepare_download.ipynb)

### 3. Download the DICOM Files

To download the scans, first install the IDC Index tool:

```
pip install idc-index
```

Then, execute the following command—replacing `<PATH_TO_MANIFEST.TXT>` and `<DOWNLOAD_DIR>` with your manifest file path and desired download directory:

```
idc download-from-manifest --manifest-file <PATH_TO_MANIFEST.TXT> --download-dir <DOWNLOAD_DIR>
```

This command downloads all the specified DICOM files into the designated directory.

### 4. Sort and Convert the Data

The downloaded data is in DICOM format. To prepare it for your experiments, follow these steps:

- **Sorting:** Organize the DICOM files using the tool "thedicomsort". While the specific usage may depend on your environment, a common workflow involves running a command to categorize files by patient or study. For example, you might first list the files and then run:
  
  ```
  thedicomsort --input <DOWNLOAD_DIR> --output <SORTED_DIR>
  ```
  
  For more detailed instructions and options, please refer to the [thedicomsort GitHub repository](https://github.com/your-repo/thedicomsort) or search online for "thedicomsort usage".

- **Conversion:** Convert the sorted DICOM files to NRRD format using Plastimatch. A typical command looks similar to:
  
  ```
  plastimatch convert --input <SORTED_DIR> --output <CONVERTED_DIR> --format nrrd
  ```
  
  For additional details and advanced options, consult the [Plastimatch documentation](http://plastimatch.org) or relevant online resources.

- **Packaging:** Finally, generate a `.pkl` file that lists the scans. This file serves as the required input for the pre-training experiments.

For a complete example of these final steps, refer again to the [prepare_download.ipynb](../../notebooks/data-download/prepare_download.ipynb) notebook.

Following these instructions will replicate the data download and preprocessing pipeline used in our study, enabling you to work with the same CT scan dataset.
