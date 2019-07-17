library(mclust)
library(rjson)

main <- function() {
	args = commandArgs(trailingOnly=TRUE)
	json_file <- args[1]
	config <- fromJSON(file=json_file)

	data <- read.csv(config$components_file, row.names=1)
	outliers <- config$outliers
	data_clean <- data[!(rownames(data) %in% outliers), ]
	print(c("N samples ", nrow(data_clean)))

	fit_gmm <- Mclust(data_clean, G=1:60, modelNames="VVI", verbose=TRUE)
	print(c("Best G ", fit_gmm$G))

	write.csv(fit_gmm$z, config$tau_file)
	write.csv(fit_gmm$classification, config$labels_file)
	write.csv(fit_gmm$BIC[,"VVI"], config$bic_file)
}

mclust.options(subset=4000)
main()