# NLP_Midterm_Osvaldo_Yizhou_David
NLP Zero-shot Self-training


Assuming that each self-training pipeline has a average of 5 steps, we would have 5 train/evaluate configuration files for each step.

For a self-training pipeline we would first have the model, apply it to unlabeled data, select 1/5th of the most confident predictions, retrain the model, get fine-tuned results for each iteration. 

In all these iterations, we would produce add peusdo labels and create a new dataset for our most confident texts/labels. This will produce 5 data sets of peudo-label datasets.

The previously defined  5 train/evaluate configuration files will have the same model condifguration since we are assuming that we can't save the model weights in NLP scholoar for each finetune interation. That is we are finetuning the same model from scratch in each iteration using the augumented dataset from the previous iteration.


In summary we would have 1 NlP base case dataset, 5 configuration file and 5 pesudo label dataset. We will evluate the final outputed pesudo-label dataset which has the label for all text as the final accuarcy of that model.

NLP_basecase_file -> used to evalate model without self-training and also used as the first stage of self-training

configuration_file -> train a model on the previous pesudo labels and generate new pesduo labels for the remaining texts and take the most confident cases

pesudo-label dataset-> used to finetune the loaded model in each stage. The final stage would create a pesudo-label dataset that has all the texts and their predicted labels. This is the dataset we will evaluate the overall accuracy on.


