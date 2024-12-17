import pandas as pd

file_path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
df = pd.read_csv(file_path)

target_features = ['fad8_chol']  # Add target features if they exist

feature_groups = {
    "subset_general": [
        'MOMID', 'diet', 'genotype', 'group', 'rs174550', 
        'OIL_CONS_AIMED', 'OIL_CONS_AIMED_cat', 'OIL_CONS_REAL'
    ] + target_features,

    "subset_health_start": [
        'fad0_weight', 'fad0_height', 'fad0_bmi', 'fad0_waist',
        'fad0_systbp1', 'fad0_diastbp1', 'fad0_systbp2', 'fad0_diastbp2',
        'fad0_systbp3', 'fad0_diastbp3', 'fad0_systbp', 'fad0_diastbp',
        'fad0_chol', 'fad0_ldlc', 'fad0_hdlc', 'fad0_tg', 'fad0_alt', 
        'fad0_creat', 'fad0_crp', 'fad0_gluc0', 'fad0_gluc30', 'fad0_gluc120',
        'fad0_hba1c', 'fad0_ins0', 'fad0_ins30', 'fad0_ins120', 'fad0_glauc'
    ] + target_features,

    "subset_biomarkers_start": [
        'fad0_gliauc', 'fad0_insauc', 'fad0_insiauc', 'fad0_matsuda', 
        'fad0_di', 'fad0_homair', 'fad0_homais', 'fad0_insgenin', 'fad0_inssec30',
        'Lnfad0_crp', 'Lnfad8_crp', 'Lnfad0_di', 'Lnfad8_di', 'FCfad_crp'
    ] + target_features,

    "subset_macronutrients_start": [
        'fad0_ENERGY_KCAL', 'fad0_ENERGY_J', 'fad0_PROTEIN', 'fad0_PROTEIN_Epros', 
        'fad0_CARBOHYDRATES', 'fad0_CARBOHYDRATES_Epros', 'fad0_ALCOHOL', 
        'fad0_ALCOHOL_Epros', 'fad0_ASH', 'fad0_FAT', 'fad0_FAT_Epros', 'fad0_FATTRI',
        'fad0_SFA', 'fad0_SFA_Epros', 'fad0_MUFA', 'fad0_MUFA_Epros', 'fad0_PUFA',
        'fad0_PUFA_Epros', 'fad0_N3_FA', 'fad0_N6_FA', 'fad0_CHOLESTEROL'
    ] + target_features,

    "subset_micronutrients_start": [
        'fad0_F16P0T', 'fad0_FA18', 'fad0_F18P1T', 'fad0_ALFALINOLENICACID', 
        'fad0_LINOLEICACID', 'fad0_EPA', 'fad0_DHA', 'fad0_F20D4N6', 'fad0_FATRN',
        'fad0_STERT', 'fad0_LACS', 'fad0_SUCROSE', 'fad0_SUGAR', 'fad0_STARCH', 
        'fad0_FIBER', 'fad0_FRUS', 'fad0_FIBINS', 'fad0_PSACNCS', 'fad0_CHOCDF'
    ] + target_features,

    "subset_minerals_start": [
        'fad0_A_VITAMIN', 'fad0_D_VITAMIN', 'fad0_E_VITAMIN', 'fad0_K_VITAMIN',
        'fad0_C_VITAMIN', 'fad0_THIAMIN_B1', 'fad0_RIBOFLAVIN_B2', 'fad0_NIASIN_EQUIVALENT',
        'fad0_PYRIDOXIN_B6', 'fad0_KOBALAMIN_B12', 'fad0_CAROTENOIDS', 'fad0_BETACAROTENS', 
        'fad0_RETINOLI', 'fad0_FOLATE', 'fad0_NATRITUM', 'fad0_MAGNESIUM', 'fad0_CALSIUM',
        'fad0_IRON', 'fad0_POTASSIUM', 'fad0_CR', 'fad0_CU', 'fad0_FD', 'fad0_IODINE', 
        'fad0_MN', 'fad0_MO', 'fad0_NATRIUMCHLORIDE', 'fad0_NT', 'fad0_PHOSPHORIUM'
    ] + target_features,

    "subset_additional_minerals": [
        'fad0_SELENIUM', 'fad0_ZINC', 'OIL_CONS_AIMED', 'OIL_CONS_AIMED_cat', 
        'OIL_CONS_REAL', 'fadinterv_ENERGY_KCAL', 'fadinterv_PROTEIN_Epros',
        'fadinterv_CARBOHYDRATES_Epros', 'fadinterv_FAT_Epros', 'fadinterv_SFA_Epros',
        'fadinterv_MUFA_Epros', 'fadinterv_PUFA_Epros', 'fadinterv_FIBER', 
        'fadinterv_SUCROSE', 'fadinterv_ALFALINOLENICACID', 'fadinterv_LINOLEICACID',
        'fadinterv_CHOLESTEROL', 'fad0_SUCROSE_epros', 'fadinterv_SUCROSE_epros', 
        'lg_fad0_ENERGY_KCAL', 'lg_fadinterv_ENERGY_KCAL', 'lg_fad0_SFA_Epros', 
        'lg_fadinterv_SFA_Epros', 'lg_fad0_MUFA_Epros', 'lg_fadinterv_MUFA_Epros'
    ] + target_features,

    "subset_changes_groups": [
        'lg_fad0_PUFA_Epros', 'lg_fadinterv_PUFA_Epros', 'lg_fad0_PROTEIN_Epros', 
        'lg_fadinterv_PROTEIN_Epros', 'lg_fad0_CHOLESTEROL', 'lg_fadinterv_CHOLESTEROL', 
        'fad0_ALCOHOL_Epros'
    ] + target_features
}

output_dir = "data/subsets/"
for subset_name, features in feature_groups.items():
    # Select columns that exist in the dataset
    selected_columns = [col for col in features if col in df.columns]
    missing_columns = set(features) - set(df.columns)
    
    if missing_columns:
        print(f"Warning: {subset_name} has missing columns: {missing_columns}")
    
    subset_df = df[selected_columns]
    subset_file = f"{output_dir}{subset_name}.csv"
    subset_df.to_csv(subset_file, index=False)
    print(f"Saved {subset_name} to {subset_file}")