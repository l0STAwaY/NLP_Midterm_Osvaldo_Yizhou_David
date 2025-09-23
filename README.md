# NLP_Midterm_Osvaldo_Yizhou_David
NLP Zero-shot Self-training


BaseCase_nlp_schloar -> The text data

Assuming that each self-training pipeline has a average of 5 steps, we would have 5 train/evaluate configuration files for each step.

For a self-training pipeline we would first have the model, apply it to unlabeled data, select 1/5th of the most confident predictions, retrain the model, get fine-tuned results for each iteration. 

In all these iterations, we would produce add peusdo labels and create a new dataset for our most confident texts/labels. This will produce 5 data sets of peudo-label datasets.

The previously defined  5 train/evaluate configuration files will have the same model condifguration since we are assuming that we can't save the model weights in NLP scholoar for each finetune interation. That is we are finetuning the same model from scratch in each iteration using the augumented dataset from the previous iteration.


In summary we would have 1 NlP base case dataset, 5 configuration file and 5 pesudo label dataset. We will evluate the final outputed pesudo-label dataset which has the label for all text as the final accuarcy of that model.

