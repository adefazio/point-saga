
echo "Running training"
#L2-regularized L1-loss support vector classification (dual)
train -v 2 -s 3 -c 1 rcv1_train.binary rcv1_ll_s3.model

#with s3: Cross Validation Accuracy = 96.6061%
#with s0 logistic: Cross Validation Accuracy = 95.9194%

echo "Testing"
#predict rcv1_test.binary rcv1_ll_s3.model rcv1_test_ll_s3.pred