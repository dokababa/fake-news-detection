# Fake-News Detection · Naïve Bayes vs. ANN

*November 2024 &nbsp;·&nbsp; Divya Gunjan*

| Model | Test accuracy |
|-------|---------------|
| Naïve Bayes | **91.1 %** |
| 2-layer ANN | 63.9 % |

## Reproduce

```r
# install deps
source("requirements.R")

# run pipeline
source("scripts/fake_news_nb_ann.R")
