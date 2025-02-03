WITH
 idc_instances_per_series AS (
 SELECT
   SeriesInstanceUID,
   COUNT(DISTINCT(SOPInstanceUID)) AS num_instances,
   COUNT(DISTINCT(ARRAY_TO_STRING(ImagePositionPatient, "/"))) AS position_count,
   MAX(SAFE_CAST(SliceThickness AS float64)) AS max_SliceThickness,
   MIN(SAFE_CAST(SliceThickness AS float64)) AS min_SliceThickness,
   STRING_AGG(DISTINCT(SAFE_CAST("LOCALIZER" IN UNNEST(ImageType) AS string)), "") AS has_localizer
 FROM
   `bigquery-public-data.idc_v14.dicom_all`
 WHERE
   Modality = "CT" and
   access = "Public"
 GROUP BY
   SeriesInstanceUID)
SELECT
 dicom_all.SeriesInstanceUID,
 ANY_VALUE(dicom_all.collection_id) as collection_id,
 ANY_VALUE(dicom_all.PatientID) AS PatientID,
 CONCAT('cp ',ANY_VALUE(dicom_all.aws_url),'.') as cp_command


FROM
 `bigquery-public-data.idc_v14.dicom_all` AS dicom_all
JOIN
 idc_instances_per_series
ON
 dicom_all.SeriesInstanceUID = idc_instances_per_series.SeriesInstanceUID
WHERE
 idc_instances_per_series.min_SliceThickness >= 1
 AND idc_instances_per_series.max_SliceThickness <= 5
 AND idc_instances_per_series.num_instances > 50
 AND idc_instances_per_series.num_instances/idc_instances_per_series.position_count = 1
 AND has_localizer = "false"
GROUP BY
 SeriesInstanceUID