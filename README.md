# Project for EECS 595

### We will explore the following:

1. Grounding to External Knowledge: Grounding  the  model  to  an  existing  Commonsense Knowledge Base such as ATOMIC (Sap et al., 2019) or ConceptNet (Speer et al.,2017). Use a Graph Attention Networks to perform message passing over a subgraph of relevant entites and then use the node representations for predictions. 


2. Improving   Entity   State   Classification: to implement a classifier based on the Recurrent Entity Network (EntNet) (Henaff et al., 2017). Alternatively, providing the full story context to BERT mayimprove results by providing a richer contex-tual embedding if the EntNet approach fails.



## How to run code
### Grounding to External Knowledge


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
