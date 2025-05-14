###############################################################################
# requirements.R  –  one-shot installer for Fake-News project
###############################################################################

cran_pkgs <- c(
  # Data wrangling / utilities
  "tidyverse", "here", "glue",
  # Text-mining stack
  "tm", "SnowballC", "gmodels",
  # Classical ML
  "e1071",
  # Deep-learning (installs TensorFlow automatically)
  "keras3"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org")
}

invisible(lapply(cran_pkgs, install_if_missing))

# keras3 will prompt to install TensorFlow the first time you call keras / tf
message("\n✔  Packages ready.  If this is your first keras3 run, call:")
message("   library(keras3); install_tensorflow()")