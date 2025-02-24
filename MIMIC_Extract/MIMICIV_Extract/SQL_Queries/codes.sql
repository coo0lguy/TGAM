SET SEARCH_PATH TO public,mimiciv_icu,mimiciv_hosp;
SELECT
    i.stay_id, d.subject_id, d.hadm_id,
    array_agg(d.icd_code ORDER BY seq_num ASC) AS icd9_codes
FROM diagnoses_icd d 
    LEFT OUTER JOIN (SELECT ccs_matched_id, icd9_code from ccs_dx) c
    ON c.icd9_code = d.icd_code
    INNER JOIN icustays i
    ON i.hadm_id = d.hadm_id AND i.subject_id = d.subject_id
WHERE d.hadm_id IN ('{hadm_id}') AND seq_num IS NOT NULL
GROUP BY i.stay_id, d.subject_id, d.hadm_id
