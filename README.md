# Project for EECS 595

### We have explored the following:

1. Commonsense contextual embeddings using COMET (Bosselut et al., 2019). 
2. Transfer learning from a related task, namely Goal-Step reasoning on the WikiHow dataset (Zhang et al., 2020).
3. Improving   Entity   State   Classification: to implement a classifier based on the Recurrent Entity Network (EntNet) (Henaff et al., 2017).



## How to run code
### Commonsense contextual embeddings and WikiHow
To run the code with the reshaped embeddings as outlined in the report, it's a three step process. 
- `git clone https://github.com/mukhal/tiered-reasoning.git`
- `git checkout main`
- Now just run the code in the `Verifiable-Coherent-NLU-COMET.ipynb` notebook to run the pipeline with COMET embeddings or `Verifiable-Coherent-NLU-Wikihow.ipynb` to run the pipeline with RoBERTa or BERT pre-trained on the goal inference task.

### Improving Entity State Classification
There were two separate evaluations of EntNet in the report.

1. Reshaped Embeddings

To run the code with the reshaped embeddings as outlined in the report, it's a three step process. 
- `git clone https://github.com/mukhal/tiered-reasoning.git`
- `git checkout entnet`
- Now just run the code in the `Verifiable-Coherent-NLU.ipynb` notebook

2. Original Embeddings

To run the code with original embeddings, with a view passed to the EntNet heads, it is the same as above but with a different branch.

To run the code with the reshaped embeddings as outlined in the report, it's a three step process. 
- `git clone https://github.com/mukhal/tiered-reasoning.git`
- `git checkout entnet-new-embedding`
- Now just run the code in the `Verifiable-Coherent-NLU.ipynb` notebook

The naming of `entnet-new-embedding` is because this was tested second, but it uses the same shape of inputs as the original TRIP baseline. Slightly confusing, but it is because it's the new embedding for the EntNet test.
