
select subject_id,
  hadm_id,
  stay_id,
  intime,
  outtime,
  age,
  gender,
  is_male,
  race,
  race_white,
  race_black,
  race_yellow,
  race_hispanic,
  race_other,
  weight,
  admission_type,
  deathtime,
  icu_los,
  hosp_los,
  mort_icu,
  mort_hosp,
  thirtyday_expire_flag,
  congestive_heart_failure,
  cardiac_arrhythmias, 
  valvular_disease, 
  pulmonary_circulation, 
  peripheral_vascular, 
  hypertension,
  paralysis,
  other_neurological,
  chronic_pulmonary,
  diabetes_uncomplicated,
  diabetes_complicated,
  hypothyroidism,
  renal_failure,
  liver_disease,
  peptic_ulcer,
  aids,
  lymphoma,
  metastatic_cancer,
  solid_tumor,
  rheumatoid_arthritis,
  coagulopathy,
  obesity,
  weight_loss,
  fluid_electrolyte,
  blood_loss_anemia,
  deficiency_anemias,
  alcohol_abuse,
  drug_abuse,
  psychoses,
  depression
  
from mimiciv_derived.new_sepsis3
where excluded=0

-- select distinct
--     i.subject_id,
--     i.hadm_id,
--     i.icustay_id,
--     i.gender,
--     i.admission_age as age,
--     i.ethnicity,
--     i.hospital_expire_flag,
--     i.hospstay_seq,
--     i.los_icu,
--     i.admittime,
--     i.dischtime,
--     i.intime,
--     i.outtime,
--     a.diagnosis AS diagnosis_at_admission,
--     a.admission_type,
--     a.insurance,
--     a.deathtime,
--     a.discharge_location,
--     CASE when a.deathtime between i.intime and i.outtime THEN 1 ELSE 0 END AS mort_icu,
--     CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
--     s.first_careunit,
--     c.fullcode_first,
--     c.dnr_first,
--     c.fullcode,
--     c.dnr,
--     c.dnr_first_charttime,
--     c.cmo_first,
--     c.cmo_last,
--     c.cmo,
--     c.timecmo_chart,
--     sofa.sofa,
--     sofa.respiration as sofa_,
--     sofa.coagulation as sofa_,
--     sofa.liver as sofa_,
--     sofa.cardiovascular as sofa_,
--     sofa.cns as sofa_,
--     sofa.renal as sofa_,
--     sapsii.sapsii,
--     sapsii.sapsii_prob, 
--     oasis.oasis,
--     oasis.oasis_prob,
--     COALESCE(f.readmission_30, 0) AS readmission_30
-- FROM icustay_detail i
--     INNER JOIN admissions a ON i.hadm_id = a.hadm_id
--     INNER JOIN icustays s ON i.icustay_id = s.icustay_id
--     INNER JOIN code_status c ON i.icustay_id = c.icustay_id
--     LEFT OUTER JOIN (SELECT d.icustay_id, 1 as readmission_30
--               FROM icustays c, icustays d
--               WHERE c.subject_id=d.subject_id
--               AND c.icustay_id > d.icustay_id
--               AND c.intime - d.outtime <= interval '30 days'
--               AND c.outtime = (SELECT MIN(e.outtime) from icustays e 
--                                 WHERE e.subject_id=c.subject_id
--                                 AND e.intime>d.outtime)) f
--               ON i.icustay_id=f.icustay_id
--     LEFT OUTER JOIN (SELECT icustay_id, sofa,  respiration, coagulation, liver, cardiovascular, cns, renal 
--               FROM sofa) sofa
--               ON i.icustay_id=sofa.icustay_id
--     LEFT OUTER JOIN (SELECT icustay_id, sapsii,  sapsii_prob 
--                 FROM sapsii) sapsii
--                 ON sapsii.icustay_id=i.icustay_id
--     LEFT OUTER JOIN (SELECT icustay_id, oasis, oasis_prob
--                 FROM oasis) oasis
--                 ON oasis.icustay_id=i.icustay_id
--     right join t1 on i.icustay_id = t1.icustay_id
-- -- WHERE --s.first_careunit NOT like 'NICU' and 
-- --     i.hadm_id is not null and i.icustay_id is not null
-- --     and i.hospstay_seq = 1
-- --     and i.icustay_seq = 1
-- --     and i.admission_age >= 16
-- --     and i.los_icu >= 1
-- --     and (i.outtime >= (i.intime + interval '24 hours'))
-- --     and (i.outtime <= (i.intime + interval '240 hours'))
-- ORDER BY subject_id

