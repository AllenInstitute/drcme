library(mclust)
library(fpc)
library(rjson)

main <- function() {
	args = commandArgs(trailingOnly=TRUE)
	json_file <- args[1]
	config <- fromJSON(file=json_file)

	merge_file <- config$merge_info_file
	tau_file <- config$post_merge_tau_file
	labels_file <- config$post_merge_labels_file

	merge_info <- fromJSON(file=merge_file)

	data <- read.csv(config$components_file, row.names=1)
	outliers <- config$outliers
	data_clean = data [!(rownames(data) %in% outliers), ]
	print(nrow(data_clean))

	gmm_fit <- Mclust(data_clean, G=merge_info$gmm_components, modelNames="VVI", verbose=TRUE)
	combi <- clustCombi(gmm_fit)

	write.csv(combi$combiz[[merge_info$postmerge_clusters]], tau_file)
	write.csv(combi$classification[[merge_info$postmerge_clusters]], labels_file)
}

mclust.options(subset=4000)
main()
