library(lavaan)
library(arrow)

data_path = 
cols = c("cbcl_scr_syn_anxdep_r", "cbcl_scr_syn_withdep_r", "cbcl_scr_syn_somatic_r", "cbcl_scr_syn_social_r", "cbcl_scr_syn_thought_r", "cbcl_scr_syn_attention_r", "cbcl_scr_syn_rulebreak_r", "cbcl_scr_syn_aggressive_r")

# ABCD
cbcl = read_parquet(paste0(data_path, "rds4_full_table_v82.parquet"))
data = cbcl[, c("subjectkey", "eventname", "site_id_l", cols)]
data = data[data$eventname == "baseline_year_1_arm_1",]

temp = data[apply(is.na(data), 1, any), c("subjectkey", "site_id_l")]
temp[, c("pf10_lavaan_cbcl_p", "pf10_lavaan_cbcl_int", "pf10_lavaan_cbcl_ext")] = NA
data = na.omit(data)
u = sort(unique(data$site_id_l))
nFold = length(u)

bifactor = '
p =~ cbcl_scr_syn_anxdep_r + cbcl_scr_syn_withdep_r + cbcl_scr_syn_somatic_r + cbcl_scr_syn_social_r + cbcl_scr_syn_thought_r + cbcl_scr_syn_attention_r + cbcl_scr_syn_rulebreak_r + cbcl_scr_syn_aggressive_r
int =~ cbcl_scr_syn_withdep_r + cbcl_scr_syn_somatic_r + cbcl_scr_syn_anxdep_r
ext =~ 1*cbcl_scr_syn_rulebreak_r + 1*cbcl_scr_syn_aggressive_r
p ~~ 0*int
p ~~ 0*ext
int ~~ 0*ext
'

scores_all = data.frame(subjectkey = data$subjectkey, site_id_l = data$site_id_l)

sall = scale(data[, cols])
mdl = cfa(bifactor, data = sall)
scoresall = data.frame(lavPredict(mdl, sall))
scores_all$pf10_lavaan_cbcl_p = scoresall$p
scores_all$pf10_lavaan_cbcl_int = scoresall$int
scores_all$pf10_lavaan_cbcl_ext = scoresall$ext

write.csv(rbind(scores_all, temp), paste0(data_path, "ABCD_lavaan_cbcl_pfactor.csv"), row.names = F, quote = F, na = "NaN")

# HCPD
cbcl_hcp = read_parquet(paste0(data_path, "hcpd_v1.parquet"))
data_hcp = cbcl_hcp[, c("subjectkey", "src_subject_id", cols)]
data_hcp["cbcl_scr_syn_withdep_r"] = 999

temp_hcp = data_hcp[apply(is.na(data_hcp), 1, any), c("subjectkey", "src_subject_id")]
temp_hcp[, c("pf10_lavaan_cbcl_p", "pf10_lavaan_cbcl_int", "pf10_lavaan_cbcl_ext")] = NA
data_hcp = na.omit(data_hcp)

scores_all_hcp = data.frame(subjectkey = data_hcp$subjectkey, src_subject_id = data_hcp$src_subject_id)

sall_hcp = data.frame(scale(data_hcp[, cols]))
sall_hcp$cbcl_scr_syn_withdep_r = 0
scoresall_hcp = data.frame(lavPredict(mdl, sall_hcp))
scores_all_hcp$pf10_lavaan_cbcl_p = scoresall_hcp$p
scores_all_hcp$pf10_lavaan_cbcl_int = scoresall_hcp$int
scores_all_hcp$pf10_lavaan_cbcl_ext = scoresall_hcp$ext


write.csv(rbind(scores_all_hcp, temp_hcp), paste0(data_path, "HCPD_lavaan_cbcl_pfactor.csv"), row.names = F, quote = F, na = "NaN")
