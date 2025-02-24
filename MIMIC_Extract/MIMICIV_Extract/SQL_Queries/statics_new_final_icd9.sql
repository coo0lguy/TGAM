with t1 as
(
select * from public.new_sepsis3
where excluded=0 and icd_version=9
)
select distinct
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.gender,
    t1.is_male,
    t1.age as age,
--     i.admission_age as age,
    i.race,
    t1.race_white,
    t1.race_black,
    t1.race_yellow,
    t1.race_hispanic,
    t1.race_other,
    t1.weight,
    ouf.urineoutput,
    gcs.gcs_eyes,
    gcs.gcs_verbal,
    gcs.gcs_motor,
    mimiciv_derived.datetime_diff(s.intime,i.admittime,'MINUTE')/60 AS hosp_adm_time,
    i.hospital_expire_flag,
--     i.hospstay_seq,
--     i.los_icu,
    t1.icu_los,
    t1.hosp_los,
    i.admittime,
    i.dischtime,
    s.intime,
    s.outtime,
    --a.diagnosis AS diagnosis_at_admission,
    a.admission_type,
    a.insurance,
    a.deathtime,
    a.discharge_location,
    t1.mort_icu,
    t1.mort_hosp,
    s.first_careunit,
--     c.fullcode_first,
--     c.dnr_first,
--     c.fullcode,
--     c.dnr,
--     c.dnr_first_charttime,
--     c.cmo_first,
--     c.cmo_last,
--     c.cmo,
--     c.timecmo_chart,
--     sofa.sofa_24hours,
--     sofa.respiration as sofa_,
--     sofa.coagulation as sofa_,
--     sofa.liver as sofa_,
--     sofa.cardiovascular as sofa_,
--     sofa.cns as sofa_,
--     sofa.renal as sofa_,
    sapsii.sapsii,
    sapsii.sapsii_prob, 
    oasis.oasis,
    oasis.oasis_prob,
    COALESCE(f.readmission_30, 0) AS readmission_30,
-- 	患者病史啥的
    t1.congestive_heart_failure,
    t1.cardiac_arrhythmias, 
    t1.valvular_disease, 
    t1.pulmonary_circulation, 
    t1.peripheral_vascular, 
    t1.hypertension,
    t1.paralysis,
    t1.other_neurological,
    t1.chronic_pulmonary,
    t1.diabetes_uncomplicated,
    t1.diabetes_complicated,
    t1.hypothyroidism,
    t1.renal_failure,
    t1.liver_disease,
    t1.peptic_ulcer,
    t1.aids,
    t1.lymphoma,
    t1.metastatic_cancer,
    t1.solid_tumor,
    t1.rheumatoid_arthritis,
    t1.coagulopathy,
    t1.obesity,
    t1.weight_loss,
    t1.fluid_electrolyte,
    t1.blood_loss_anemia,
    t1.deficiency_anemias,
    t1.alcohol_abuse,
    t1.drug_abuse,
    t1.psychoses,
    t1.depression
FROM mimiciv_derived.icustay_detail i
    INNER JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN mimiciv_icu.icustays s ON i.stay_id = s.stay_id
--     INNER JOIN mimiciii.code_status c ON i.stay_id = c.stay_id
    LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
              FROM mimiciv_icu.icustays c, mimiciv_icu.icustays d
              WHERE c.subject_id=d.subject_id
              AND c.stay_id > d.stay_id
              AND c.intime - d.outtime <= interval '30 days'
              AND c.outtime = (SELECT MIN(e.outtime) from mimiciv_icu.icustays e 
                                WHERE e.subject_id=c.subject_id
                                AND e.intime>d.outtime)) f
              ON i.stay_id=f.stay_id
--     LEFT OUTER JOIN (SELECT stay_id, sofa_24hours,  respiration, coagulation, liver, cardiovascular, cns, renal 
--               FROM mimiciv_derived.sofa) sofa
--               ON i.stay_id=sofa.stay_id
    LEFT OUTER JOIN (SELECT stay_id, sapsii,  sapsii_prob 
                FROM mimiciv_derived.sapsii) sapsii
                ON sapsii.stay_id=i.stay_id
    LEFT OUTER JOIN (SELECT stay_id, oasis, oasis_prob
                FROM mimiciv_derived.oasis) oasis
                ON oasis.stay_id=i.stay_id
    LEFT OUTER JOIN (SELECT stay_id, urineoutput
                FROM mimiciv_derived.first_day_urine_output) ouf
                ON ouf.stay_id=i.stay_id
    LEFT OUTER JOIN (SELECT stay_id, gcs_motor, gcs_verbal, gcs_eyes
                FROM mimiciv_derived.first_day_gcs) gcs
                ON gcs.stay_id=i.stay_id
    right join t1 on i.stay_id = t1.stay_id
-- WHERE --s.first_careunit NOT like 'NICU' and 
--     i.hadm_id is not null and i.icustay_id is not null
--     and i.hospstay_seq = 1
--     and i.icustay_seq = 1
--     and i.admission_age >= 16
--     and i.los_icu >= 1
--     and (i.outtime >= (i.intime + interval '24 hours'))
--     and (i.outtime <= (i.intime + interval '240 hours'))
ORDER BY subject_id
