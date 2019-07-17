library(mclust)
library(fpc)
library(rjson)

gmm_combi_clust_CBI <- function (data, G, H) {
	n <- nrow(data)
	print(c("G ", G))
	print(c("H ", H))

	gmm_fit <- Mclust(data, G=1:G, modelNames="VVI", verbose=TRUE)
#	if (is.null(gmm_fit)) {
#		gmm_fit <- Mclust(data, G=G, modelNames=c("EEI", "EVI", "VEI", "VVI"), verbose=TRUE)var(data)/G^(2/d)
#	}
	combi <- clustCombi(gmm_fit)

	nc <- H
	if (gmm_fit$G < H) {
		nc <- gmm_fit$G
	}
	print(c("n samples", n))
	print(c("selected clusters ", nc))
	clusterlist <- list()
	for (i in 1:nc) {
		clusterlist[[i]] <- (combi$classification[[nc]] == i)
	}
	out <- list(result = combi,
				nc = nc,
				clusterlist = clusterlist,
				partition = rep(1, n),
				clustermethod = "gmm_combi_clust_CBI")
}

gmm_clust_CBI <- function (data, G) {
	n <- nrow(data)
	gmm_fit <- Mclust(data, G=G, modelNames="VVI", verbose=TRUE)

	nc <- gmm_fit$G
	print(n)
	print(nc)
	clusterlist <- list()
	for (i in 1:nc) {
		clusterlist[[i]] <- (gmm_fit$classification == i)
	}
	out <- list(result = gmm_fit,
				nc = nc,
				clusterlist = clusterlist,
				partition = rep(1, n),
				clustermethod = "gmm_clust_CBI")
}

main <- function() {
	args = commandArgs(trailingOnly=TRUE)
	json_file <- args[1]
	config <- fromJSON(file=json_file)

	merge_file <- config$merge_info_file
	jaccard_file <- config$jaccard_file
	merge_info <- fromJSON(file=merge_file)

	data <- read.csv(config$components_file, row.names=1)
	outliers <- config$outliers
	data_clean = data[!(rownames(data) %in% outliers), ]
	print(nrow(data_clean))

	res <- clusterboot(data_clean, B=100, bootmethod="subset",
					   subtuning=floor(nrow(data_clean) * 0.9),
					   clustermethod=gmm_combi_clust_CBI,
					   G=merge_info$gmm_components,
					   H=merge_info$postmerge_clusters)

	print(res)
	write.csv(res$subsetmean, jaccard_file)

}

mclust.options(subset=4000)
main()
